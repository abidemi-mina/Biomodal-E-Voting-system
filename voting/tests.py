"""
tests.py
========
Test suite for the NigeriaVotes E-Voting System.

Run:
    python manage.py test voting
    # or with coverage:
    coverage run manage.py test voting && coverage report
"""

import io
import json
import pickle
import hashlib
import secrets
import numpy as np
from datetime import date, timedelta
from unittest.mock import patch, MagicMock
from PIL import Image as PILImage

from django.test import TestCase, Client
from django.urls import reverse
from django.utils import timezone
from django.contrib.auth.models import User

from voting.models import Voter, Election, Candidate, Vote, AuditLog, VoterSession
from voting.biometrics import (
    cosine_similarity,
    compute_metrics,
    base64_to_bytes,
    BiometricVerifier,
)


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def make_fake_embedding(seed=42, size=512):
    rng = np.random.RandomState(seed)
    v = rng.randn(size).astype(np.float32)
    return v / np.linalg.norm(v)


def make_dummy_image_bytes():
    """Create a tiny valid JPEG in memory."""
    buf = io.BytesIO()
    img = PILImage.new('RGB', (64, 64), color=(100, 150, 200))
    img.save(buf, format='JPEG')
    return buf.getvalue()


def create_voter(voter_id='NIN0001234567', **kwargs):
    emb = make_fake_embedding(seed=1)
    return Voter.objects.create(
        voter_id=voter_id,
        full_name=kwargs.get('full_name', 'Adaeze Okafor'),
        email=kwargs.get('email', f'{voter_id}@test.ng'),
        phone='+2348012345678',
        date_of_birth=date(1990, 5, 15),
        gender='F',
        state='Anambra',
        lga='Awka South',
        ward='Ward 3',
        polling_unit='PU 005',
        face_encoding=pickle.dumps(emb),
        status='approved',
    )


def create_election(status='active', **kwargs):
    now = timezone.now()
    return Election.objects.create(
        title=kwargs.get('title', '2027 Test Election'),
        election_type=kwargs.get('election_type', 'presidential'),
        start_date=now - timedelta(hours=1),
        end_date=now + timedelta(hours=23),
        status=status,
    )


def create_candidate(election, **kwargs):
    return Candidate.objects.create(
        election=election,
        full_name=kwargs.get('full_name', 'Emeka Test'),
        party=kwargs.get('party', 'APC'),
        position_number=kwargs.get('position_number', 1),
    )


def create_voter_session(voter):
    token = secrets.token_hex(32)
    VoterSession.objects.filter(voter=voter).delete()
    return VoterSession.objects.create(
        voter=voter,
        session_token=token,
        expires_at=timezone.now() + timedelta(hours=1),
        face_confidence=0.95,
        liveness_score=0.92,
    )


# ─────────────────────────────────────────────────────────────
# 1. Model Tests
# ─────────────────────────────────────────────────────────────
class VoterModelTest(TestCase):
    def test_create_voter(self):
        v = create_voter()
        self.assertEqual(v.status, 'approved')
        self.assertEqual(v.state, 'Anambra')
        self.assertIsNotNone(v.face_encoding)

    def test_voter_str(self):
        v = create_voter()
        self.assertIn('NIN0001234567', str(v))

    def test_voter_id_unique(self):
        create_voter('UNIQUE001')
        with self.assertRaises(Exception):
            create_voter('UNIQUE001')  # duplicate

    def test_voter_email_unique(self):
        create_voter('ID001', email='same@test.ng')
        with self.assertRaises(Exception):
            create_voter('ID002', email='same@test.ng')


class ElectionModelTest(TestCase):
    def test_create_election(self):
        e = create_election()
        self.assertEqual(e.status, 'active')
        self.assertTrue(e.is_active)

    def test_total_votes(self):
        e = create_election()
        c = create_candidate(e)
        v = create_voter()
        voter_hash = hashlib.sha256(f"{v.voter_id}:{e.id}:evoting_salt_2024".encode()).hexdigest()
        Vote.objects.create(election=e, candidate=c, voter_hash=voter_hash)
        self.assertEqual(e.total_votes, 1)

    def test_election_str(self):
        e = create_election(title='Presidential Test')
        self.assertIn('Presidential Test', str(e))


class CandidateModelTest(TestCase):
    def test_vote_count(self):
        e = create_election()
        c1 = create_candidate(e, party='APC', position_number=1)
        c2 = create_candidate(e, party='PDP', position_number=2)
        v = create_voter()
        voter_hash = hashlib.sha256(f"{v.voter_id}:{e.id}:evoting_salt_2024".encode()).hexdigest()
        Vote.objects.create(election=e, candidate=c1, voter_hash=voter_hash)
        self.assertEqual(c1.vote_count, 1)
        self.assertEqual(c2.vote_count, 0)

    def test_vote_percentage(self):
        e = create_election()
        c1 = create_candidate(e, party='APC', position_number=1)
        c2 = create_candidate(e, party='PDP', position_number=2)
        for i in range(3):
            v = create_voter(voter_id=f'V00{i}', email=f'v{i}@test.ng')
            vh = hashlib.sha256(f"{v.voter_id}:{e.id}:evoting_salt_2024".encode()).hexdigest()
            Vote.objects.create(election=e, candidate=c1, voter_hash=vh)
        v4 = create_voter(voter_id='V003', email='v3@test.ng')
        vh4 = hashlib.sha256(f"{v4.voter_id}:{e.id}:evoting_salt_2024".encode()).hexdigest()
        Vote.objects.create(election=e, candidate=c2, voter_hash=vh4)
        self.assertAlmostEqual(c1.vote_percentage, 75.0)
        self.assertAlmostEqual(c2.vote_percentage, 25.0)


class AuditLogModelTest(TestCase):
    def test_create_audit_log(self):
        log = AuditLog.objects.create(
            action='auth_success',
            voter_id='NIN001',
            description='Test auth success',
            ip_address='127.0.0.1',
        )
        self.assertTrue(log.success)
        self.assertEqual(log.action, 'auth_success')


# ─────────────────────────────────────────────────────────────
# 2. Biometric Engine Tests
# ─────────────────────────────────────────────────────────────
class CosineSimilarityTest(TestCase):
    def test_identical_vectors(self):
        v = make_fake_embedding(seed=1)
        sim = cosine_similarity(v, v)
        self.assertAlmostEqual(sim, 1.0, places=5)

    def test_opposite_vectors(self):
        v = make_fake_embedding(seed=1)
        sim = cosine_similarity(v, -v)
        self.assertAlmostEqual(sim, -1.0, places=5)

    def test_orthogonal_vectors(self):
        a = np.zeros(4, dtype=np.float32)
        b = np.zeros(4, dtype=np.float32)
        a[0] = 1.0
        b[1] = 1.0
        sim = cosine_similarity(a, b)
        self.assertAlmostEqual(sim, 0.0, places=5)

    def test_similar_vectors(self):
        rng = np.random.RandomState(0)
        v = rng.randn(512).astype(np.float32)
        noise = rng.randn(512).astype(np.float32) * 0.05
        v2 = v + noise
        sim = cosine_similarity(v / np.linalg.norm(v), v2 / np.linalg.norm(v2))
        self.assertGreater(sim, 0.97)

    def test_zero_vector(self):
        v = make_fake_embedding(seed=1)
        z = np.zeros(512, dtype=np.float32)
        sim = cosine_similarity(v, z)
        self.assertEqual(sim, 0.0)


class ComputeMetricsTest(TestCase):
    def test_perfect_system(self):
        true_labels = [1, 1, 1, 0, 0, 0]
        scores =      [0.95, 0.96, 0.94, 0.3, 0.2, 0.1]
        m = compute_metrics(true_labels, scores, threshold=0.85)
        self.assertEqual(m['accuracy'], 100.0)
        self.assertEqual(m['far'], 0.0)
        self.assertEqual(m['frr'], 0.0)

    def test_all_wrong(self):
        true_labels = [1, 1, 0, 0]
        scores =      [0.2, 0.3, 0.95, 0.96]  # All wrong
        m = compute_metrics(true_labels, scores, threshold=0.85)
        self.assertEqual(m['tp'], 0)
        self.assertEqual(m['tn'], 0)

    def test_far_frr(self):
        true_labels = [1, 1, 1, 1, 0, 0, 0, 0]
        # 1 genuine rejected (FN), 1 impostor accepted (FP)
        scores = [0.9, 0.9, 0.9, 0.5, 0.9, 0.2, 0.2, 0.2]
        m = compute_metrics(true_labels, scores, threshold=0.85)
        self.assertEqual(m['fn'], 1)
        self.assertEqual(m['fp'], 1)
        self.assertGreater(m['frr'], 0)
        self.assertGreater(m['far'], 0)


class BiometricVerifierTest(TestCase):
    """Tests for BiometricVerifier in stub mode (no torch installed)."""

    def setUp(self):
        self.verifier = BiometricVerifier()

    def test_register_no_images(self):
        result = self.verifier.register([])
        self.assertFalse(result['success'])

    def test_register_success_stub(self):
        """In stub mode, register should succeed with any image bytes."""
        img_bytes = make_dummy_image_bytes()
        result = self.verifier.register([img_bytes] * 5)
        # In stub mode FaceNetEncoder returns random embedding
        self.assertTrue(result['success'])
        self.assertIsNotNone(result['embedding'])

    def test_authenticate_stub(self):
        """In stub mode, authentication should succeed (stub liveness=True, similarity computed)."""
        img_bytes = make_dummy_image_bytes()
        reg = self.verifier.register([img_bytes] * 3)
        self.assertTrue(reg['success'])
        auth = self.verifier.authenticate(img_bytes, reg['embedding'])
        self.assertIn('authenticated', auth)
        self.assertIn('liveness_pass', auth)
        self.assertIn('liveness_score', auth)
        self.assertIn('similarity', auth)


# ─────────────────────────────────────────────────────────────
# 3. View Tests
# ─────────────────────────────────────────────────────────────
class HomeViewTest(TestCase):
    def test_home_loads(self):
        resp = self.client.get('/')
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, 'NigeriaVotes')

    def test_results_list_loads(self):
        resp = self.client.get('/results/')
        self.assertEqual(resp.status_code, 200)


class VoterRegistrationViewTest(TestCase):
    def test_register_page_loads(self):
        resp = self.client.get('/register/')
        self.assertEqual(resp.status_code, 200)

    def test_register_post_valid(self):
        resp = self.client.post('/register/', {
            'voter_id': 'TEST00001',
            'full_name': 'Kunle Adeyemi',
            'email': 'kunle@test.ng',
            'confirm_email': 'kunle@test.ng',
            'phone': '+2348012345678',
            'date_of_birth': '1985-03-20',
            'gender': 'M',
            'state': 'Oyo',
            'lga': 'Ibadan North',
            'ward': 'Ward 5',
            'polling_unit': 'PU 010',
        })
        # Should redirect to face capture
        self.assertEqual(resp.status_code, 302)
        self.assertRedirects(resp, '/register/face/')

    def test_register_post_underage(self):
        resp = self.client.post('/register/', {
            'voter_id': 'TEST00002',
            'full_name': 'Young Person',
            'email': 'young@test.ng',
            'confirm_email': 'young@test.ng',
            'phone': '+2348012345678',
            'date_of_birth': '2015-01-01',  # Under 18
            'gender': 'M',
            'state': 'Lagos',
            'lga': 'Eti-Osa',
            'ward': 'Ward 1',
            'polling_unit': 'PU 001',
        })
        self.assertEqual(resp.status_code, 200)  # Form re-rendered with error

    def test_register_face_requires_session(self):
        # Without pending_registration in session
        resp = self.client.get('/register/face/')
        self.assertRedirects(resp, '/register/')


class VoterLoginViewTest(TestCase):
    def setUp(self):
        self.voter = create_voter()

    def test_login_page_loads(self):
        resp = self.client.get('/login/')
        self.assertEqual(resp.status_code, 200)

    def test_login_valid_voter(self):
        resp = self.client.post('/login/', {'voter_id': self.voter.voter_id})
        self.assertRedirects(resp, '/authenticate/')
        self.assertEqual(self.client.session['auth_voter_id'], self.voter.voter_id)

    def test_login_invalid_voter(self):
        resp = self.client.post('/login/', {'voter_id': 'NONEXISTENT999'})
        self.assertEqual(resp.status_code, 200)
        # Should show error message

    def test_authenticate_page_requires_session(self):
        resp = self.client.get('/authenticate/')
        self.assertRedirects(resp, '/login/')


class VotingViewTest(TestCase):
    def setUp(self):
        self.voter = create_voter()
        self.election = create_election()
        self.candidate1 = create_candidate(self.election, party='APC', position_number=1)
        self.candidate2 = create_candidate(self.election, party='PDP', position_number=2)
        # Authenticate voter
        session = create_voter_session(self.voter)
        self.client.session['voter_session_token'] = session.session_token
        self.client.session.save()
        # Patch session lookup
        s = self.client.session
        s['voter_session_token'] = session.session_token
        s.save()

    def test_vote_dashboard_loads(self):
        s = self.client.session
        vs = create_voter_session(self.voter)
        s['voter_session_token'] = vs.session_token
        s.save()
        resp = self.client.get('/vote/')
        self.assertEqual(resp.status_code, 200)

    def test_vote_dashboard_requires_auth(self):
        # New client with no session
        c = Client()
        resp = c.get('/vote/')
        self.assertRedirects(resp, '/login/')

    def test_ballot_page_loads(self):
        vs = create_voter_session(self.voter)
        s = self.client.session
        s['voter_session_token'] = vs.session_token
        s.save()
        resp = self.client.get(f'/vote/{self.election.id}/')
        self.assertEqual(resp.status_code, 200)


class APICastVoteTest(TestCase):
    def setUp(self):
        self.voter = create_voter()
        self.election = create_election()
        self.candidate = create_candidate(self.election)
        self.vs = create_voter_session(self.voter)
        s = self.client.session
        s['voter_session_token'] = self.vs.session_token
        s.save()

    def test_cast_vote_success(self):
        resp = self.client.post(
            '/api/vote/cast/',
            data=json.dumps({
                'election_id': str(self.election.id),
                'candidate_id': str(self.candidate.id),
            }),
            content_type='application/json',
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data['success'])
        self.assertEqual(Vote.objects.count(), 1)

    def test_cast_vote_duplicate(self):
        # First vote
        self.client.post(
            '/api/vote/cast/',
            data=json.dumps({
                'election_id': str(self.election.id),
                'candidate_id': str(self.candidate.id),
            }),
            content_type='application/json',
        )
        # Second vote attempt
        resp = self.client.post(
            '/api/vote/cast/',
            data=json.dumps({
                'election_id': str(self.election.id),
                'candidate_id': str(self.candidate.id),
            }),
            content_type='application/json',
        )
        data = resp.json()
        self.assertFalse(data['success'])
        self.assertEqual(Vote.objects.count(), 1)  # Still only 1 vote

    def test_cast_vote_requires_auth(self):
        c = Client()  # No session
        resp = c.post(
            '/api/vote/cast/',
            data=json.dumps({
                'election_id': str(self.election.id),
                'candidate_id': str(self.candidate.id),
            }),
            content_type='application/json',
        )
        # Should redirect or return error
        self.assertNotEqual(resp.status_code, 500)


# ─────────────────────────────────────────────────────────────
# 4. API Authentication Tests
# ─────────────────────────────────────────────────────────────
class APIAuthTest(TestCase):
    def setUp(self):
        self.voter = create_voter()
        # Set auth session
        s = self.client.session
        s['auth_voter_id'] = self.voter.voter_id
        s.save()

    def test_authenticate_no_image(self):
        resp = self.client.post(
            '/api/authenticate/',
            data=json.dumps({}),
            content_type='application/json',
        )
        data = resp.json()
        self.assertFalse(data['success'])

    def test_authenticate_missing_session(self):
        c = Client()
        import base64
        img_b64 = base64.b64encode(make_dummy_image_bytes()).decode()
        resp = c.post(
            '/api/authenticate/',
            data=json.dumps({'image': img_b64}),
            content_type='application/json',
        )
        data = resp.json()
        self.assertFalse(data['success'])


# ─────────────────────────────────────────────────────────────
# 5. Admin View Tests
# ─────────────────────────────────────────────────────────────
class AdminPanelTest(TestCase):
    def setUp(self):
        self.admin = User.objects.create_superuser('testadmin', 'admin@test.ng', 'adminpass123')
        self.client.login(username='testadmin', password='adminpass123')

    def test_admin_dashboard(self):
        resp = self.client.get('/admin-panel/')
        self.assertEqual(resp.status_code, 200)

    def test_admin_voters(self):
        create_voter()
        resp = self.client.get('/admin-panel/voters/')
        self.assertEqual(resp.status_code, 200)

    def test_admin_elections(self):
        resp = self.client.get('/admin-panel/elections/')
        self.assertEqual(resp.status_code, 200)

    def test_admin_create_election(self):
        now = timezone.now()
        resp = self.client.post('/admin-panel/elections/', {
            'title': 'Test Senate 2027',
            'election_type': 'senate',
            'description': '',
            'start_date': (now - timedelta(hours=1)).strftime('%Y-%m-%dT%H:%M'),
            'end_date': (now + timedelta(hours=10)).strftime('%Y-%m-%dT%H:%M'),
        })
        self.assertEqual(resp.status_code, 302)
        self.assertTrue(Election.objects.filter(title='Test Senate 2027').exists())

    def test_admin_metrics(self):
        resp = self.client.get('/admin-panel/metrics/')
        self.assertEqual(resp.status_code, 200)

    def test_admin_audit_logs(self):
        AuditLog.objects.create(
            action='vote_cast',
            voter_id='NIN001',
            description='Test log',
        )
        resp = self.client.get('/admin-panel/audit/')
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, 'vote_cast')

    def test_admin_requires_staff(self):
        regular = User.objects.create_user('regular', 'r@test.ng', 'pass')
        c = Client()
        c.login(username='regular', password='pass')
        resp = c.get('/admin-panel/')
        # Should redirect to home (not staff)
        self.assertRedirects(resp, '/')

    def test_approve_voter(self):
        voter = create_voter()
        voter.status = 'pending'
        voter.save()
        resp = self.client.get(f'/admin-panel/voters/{voter.id}/approve/')
        voter.refresh_from_db()
        self.assertEqual(voter.status, 'approved')

    def test_toggle_election(self):
        e = create_election(status='active')
        self.client.get(f'/admin-panel/elections/{e.id}/toggle/')
        e.refresh_from_db()
        self.assertEqual(e.status, 'completed')


# ─────────────────────────────────────────────────────────────
# 6. Results View Tests
# ─────────────────────────────────────────────────────────────
class ResultsViewTest(TestCase):
    def test_results_list(self):
        resp = self.client.get('/results/')
        self.assertEqual(resp.status_code, 200)

    def test_election_results(self):
        e = create_election()
        c1 = create_candidate(e, party='APC', position_number=1)
        c2 = create_candidate(e, party='PDP', position_number=2)
        voter = create_voter()
        vh = hashlib.sha256(f"{voter.voter_id}:{e.id}:evoting_salt_2024".encode()).hexdigest()
        Vote.objects.create(election=e, candidate=c1, voter_hash=vh)

        resp = self.client.get(f'/results/{e.id}/')
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, 'Emeka Test')

    def test_election_results_404(self):
        import uuid
        resp = self.client.get(f'/results/{uuid.uuid4()}/')
        self.assertEqual(resp.status_code, 404)


# ─────────────────────────────────────────────────────────────
# 7. Vote Integrity Tests
# ─────────────────────────────────────────────────────────────
class VoteIntegrityTest(TestCase):
    """Verifies core electoral integrity guarantees."""

    def test_one_voter_one_vote(self):
        """Same voter cannot vote twice in same election."""
        election = create_election()
        candidate = create_candidate(election)
        voter = create_voter()
        vh = hashlib.sha256(f"{voter.voter_id}:{election.id}:evoting_salt_2024".encode()).hexdigest()

        Vote.objects.create(election=election, candidate=candidate, voter_hash=vh)

        from django.db import IntegrityError
        with self.assertRaises(IntegrityError):
            Vote.objects.create(election=election, candidate=candidate, voter_hash=vh)

    def test_voter_can_vote_in_multiple_elections(self):
        """Voter can vote in different elections."""
        e1 = create_election(title='Election 1')
        e2 = create_election(title='Election 2')
        c1 = create_candidate(e1, party='APC')
        c2 = create_candidate(e2, party='PDP')
        voter = create_voter()

        vh1 = hashlib.sha256(f"{voter.voter_id}:{e1.id}:evoting_salt_2024".encode()).hexdigest()
        vh2 = hashlib.sha256(f"{voter.voter_id}:{e2.id}:evoting_salt_2024".encode()).hexdigest()

        Vote.objects.create(election=e1, candidate=c1, voter_hash=vh1)
        Vote.objects.create(election=e2, candidate=c2, voter_hash=vh2)

        self.assertEqual(Vote.objects.filter(election=e1).count(), 1)
        self.assertEqual(Vote.objects.filter(election=e2).count(), 1)

    def test_voter_hash_anonymization(self):
        """Voter hash doesn't directly expose voter_id."""
        voter_id = 'NIN0001234567'
        election_id = 'some-election-id'
        vh = hashlib.sha256(f"{voter_id}:{election_id}:evoting_salt_2024".encode()).hexdigest()
        self.assertNotIn(voter_id, vh)
        self.assertNotIn(election_id, vh)
        self.assertEqual(len(vh), 64)  # SHA-256 hex = 64 chars

    def test_different_voters_different_hashes(self):
        """Two different voters get different hashes for same election."""
        e = create_election()
        vh1 = hashlib.sha256(f"VOTER001:{e.id}:evoting_salt_2024".encode()).hexdigest()
        vh2 = hashlib.sha256(f"VOTER002:{e.id}:evoting_salt_2024".encode()).hexdigest()
        self.assertNotEqual(vh1, vh2)
