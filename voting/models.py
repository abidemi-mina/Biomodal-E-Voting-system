from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone
import uuid


class Voter(models.Model):
    """
    Registered voter with biometric data
    """
    GENDER_CHOICES = [('M', 'Male'), ('F', 'Female'), ('O', 'Other')]
    STATUS_CHOICES = [('pending', 'Pending'), ('approved', 'Approved'), ('rejected', 'Rejected')]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    voter_id = models.CharField(max_length=20, unique=True)
    full_name = models.CharField(max_length=200)
    email = models.EmailField(unique=True)
    phone = models.CharField(max_length=15)
    date_of_birth = models.DateField()
    gender = models.CharField(max_length=1, choices=GENDER_CHOICES)
    state = models.CharField(max_length=100)
    lga = models.CharField(max_length=100)
    ward = models.CharField(max_length=100)
    polling_unit = models.CharField(max_length=200)

    # Biometric data
    face_encoding = models.BinaryField(null=True, blank=True)  # Stored FaceNet embeddings
    face_image = models.ImageField(upload_to='voter_images/', null=True, blank=True)

    # Registration & status
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default='pending')
    registered_at = models.DateTimeField(default=timezone.now)
    is_active = models.BooleanField(default=True)

    class Meta:
        db_table = 'voters'
        ordering = ['-registered_at']

    def __str__(self):
        return f"{self.voter_id} - {self.full_name}"


class Election(models.Model):
    """
    An election event
    """
    ELECTION_TYPES = [
        ('presidential', 'Presidential'),
        ('gubernatorial', 'Gubernatorial'),
        ('senate', 'Senate'),
        ('house', 'House of Representatives'),
        ('state_assembly', 'State House of Assembly'),
        ('local', 'Local Government'),
    ]
    STATUS_CHOICES = [
        ('upcoming', 'Upcoming'),
        ('active', 'Active'),
        ('completed', 'Completed'),
        ('cancelled', 'Cancelled'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    title = models.CharField(max_length=300)
    election_type = models.CharField(max_length=20, choices=ELECTION_TYPES)
    description = models.TextField(blank=True)
    start_date = models.DateTimeField()
    end_date = models.DateTimeField()
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default='upcoming')
    created_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True)
    created_at = models.DateTimeField(default=timezone.now)

    class Meta:
        db_table = 'elections'
        ordering = ['-start_date']

    def __str__(self):
        return self.title

    @property
    def is_active(self):
        now = timezone.now()
        return self.start_date <= now <= self.end_date and self.status == 'active'

    @property
    def total_votes(self):
        return self.votes.count()


class Candidate(models.Model):
    """
    A candidate in an election
    """
    PARTY_CHOICES = [
        ('APC', 'All Progressives Congress'),
        ('PDP', 'Peoples Democratic Party'),
        ('LP', 'Labour Party'),
        ('NNPP', 'New Nigeria Peoples Party'),
        ('APGA', 'All Progressives Grand Alliance'),
        ('SDP', 'Social Democratic Party'),
        ('ADC', 'African Democratic Congress'),
        ('IND', 'Independent'),
        ('OTHER', 'Other'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    election = models.ForeignKey(Election, on_delete=models.CASCADE, related_name='candidates')
    full_name = models.CharField(max_length=200)
    party = models.CharField(max_length=10, choices=PARTY_CHOICES)
    party_name = models.CharField(max_length=100, blank=True)
    bio = models.TextField(blank=True)
    photo = models.ImageField(upload_to='candidate_photos/', null=True, blank=True)
    position_number = models.IntegerField(default=1)  # Ballot position

    class Meta:
        db_table = 'candidates'
        ordering = ['position_number']
        unique_together = ['election', 'party']

    def __str__(self):
        return f"{self.full_name} ({self.party}) - {self.election.title}"

    @property
    def vote_count(self):
        return self.votes.count()

    @property
    def vote_percentage(self):
        total = self.election.total_votes
        if total == 0:
            return 0
        return round((self.vote_count / total) * 100, 2)


class Vote(models.Model):
    """
    A cast vote - anonymized after verification
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    election = models.ForeignKey(Election, on_delete=models.CASCADE, related_name='votes')
    candidate = models.ForeignKey(Candidate, on_delete=models.CASCADE, related_name='votes')
    # voter stored separately for audit but vote itself is anonymous
    voter_hash = models.CharField(max_length=64)  # SHA-256 hash of voter_id + election_id
    voted_at = models.DateTimeField(default=timezone.now)
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    device_fingerprint = models.CharField(max_length=200, blank=True)

    class Meta:
        db_table = 'votes'
        unique_together = ['election', 'voter_hash']  # One vote per voter per election

    def __str__(self):
        return f"Vote in {self.election.title} at {self.voted_at}"


class AuditLog(models.Model):
    """
    Audit trail for all significant system actions
    """
    ACTION_TYPES = [
        ('register', 'Voter Registration'),
        ('login', 'Login Attempt'),
        ('auth_success', 'Authentication Success'),
        ('auth_fail', 'Authentication Failure'),
        ('vote_cast', 'Vote Cast'),
        ('vote_attempt', 'Vote Attempt'),
        ('liveness_fail', 'Liveness Check Failed'),
        ('admin_action', 'Admin Action'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    action = models.CharField(max_length=20, choices=ACTION_TYPES)
    voter_id = models.CharField(max_length=20, blank=True)
    election_id = models.CharField(max_length=50, blank=True)
    description = models.TextField()
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    timestamp = models.DateTimeField(default=timezone.now)
    success = models.BooleanField(default=True)

    class Meta:
        db_table = 'audit_logs'
        ordering = ['-timestamp']

    def __str__(self):
        return f"{self.action} - {self.voter_id} at {self.timestamp}"


class VoterSession(models.Model):
    """
    Tracks authenticated voter sessions for voting
    """
    voter = models.OneToOneField(Voter, on_delete=models.CASCADE, related_name='session')
    session_token = models.CharField(max_length=64, unique=True)
    authenticated_at = models.DateTimeField(default=timezone.now)
    expires_at = models.DateTimeField()
    face_confidence = models.FloatField(default=0.0)
    liveness_score = models.FloatField(default=0.0)
    is_valid = models.BooleanField(default=True)

    class Meta:
        db_table = 'voter_sessions'
