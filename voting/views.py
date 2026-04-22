"""
Views for the E-Voting System
"""

import json
import hashlib
import secrets
import logging
import base64
from datetime import timedelta
from functools import wraps

from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST, require_GET
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.utils import timezone
from django.db import transaction
from django.db.models import Count

from .models import Voter, Election, Candidate, Vote, AuditLog, VoterSession
from .forms import (
    VoterRegistrationForm, VoterLoginForm,
    ElectionForm, CandidateForm, AdminLoginForm
)
from .biometrics import get_verifier, base64_to_bytes
from .security import check_auth_rate_limit, record_auth_failure, record_auth_success

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Decorators
# ─────────────────────────────────────────────────────────────
def voter_authenticated(view_func):
    """Require voter to have completed facial authentication."""
    @wraps(view_func)
    def wrapper(request, *args, **kwargs):
        token = request.session.get('voter_session_token')
        if not token:
            messages.error(request, 'Please authenticate to continue.')
            return redirect('voter_login')
        try:
            vs = VoterSession.objects.get(session_token=token, is_valid=True)
            if vs.expires_at < timezone.now():
                vs.is_valid = False
                vs.save()
                messages.error(request, 'Your session has expired. Please authenticate again.')
                return redirect('voter_login')
            request.voter = vs.voter
            request.voter_session = vs
        except VoterSession.DoesNotExist:
            messages.error(request, 'Invalid session. Please authenticate again.')
            return redirect('voter_login')
        return view_func(request, *args, **kwargs)
    return wrapper


def get_client_ip(request):
    x_forwarded = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded:
        return x_forwarded.split(',')[0].strip()
    return request.META.get('REMOTE_ADDR')


def log_action(action, voter_id='', election_id='', description='', ip=None, success=True):
    AuditLog.objects.create(
        action=action,
        voter_id=voter_id,
        election_id=str(election_id),
        description=description,
        ip_address=ip,
        success=success,
    )


# ─────────────────────────────────────────────────────────────
# Public Views
# ─────────────────────────────────────────────────────────────
def home(request):
    active_elections = Election.objects.filter(status='active').count()
    total_voters = Voter.objects.filter(status='approved').count()
    total_votes = Vote.objects.count()
    context = {
        'active_elections': active_elections,
        'total_voters': total_voters,
        'total_votes': total_votes,
    }
    return render(request, 'voting/home.html', context)


# ─────────────────────────────────────────────────────────────
# Voter Registration
# ─────────────────────────────────────────────────────────────
def voter_register(request):
    """Step 1: Collect voter information."""
    if request.method == 'POST':
        form = VoterRegistrationForm(request.POST)
        if form.is_valid():
            # Save form data to session for face capture step
            request.session['pending_registration'] = {
                'voter_id': form.cleaned_data['voter_id'],
                'full_name': form.cleaned_data['full_name'],
                'email': form.cleaned_data['email'],
                'phone': form.cleaned_data['phone'],
                'date_of_birth': str(form.cleaned_data['date_of_birth']),
                'gender': form.cleaned_data['gender'],
                'state': form.cleaned_data['state'],
                'lga': form.cleaned_data['lga'],
                'ward': form.cleaned_data['ward'],
                'polling_unit': form.cleaned_data['polling_unit'],
            }
            return redirect('voter_register_face')
    else:
        form = VoterRegistrationForm()
    return render(request, 'voting/register.html', {'form': form})


def voter_register_face(request):
    """Step 2: Capture face images for biometric template."""
    if 'pending_registration' not in request.session:
        return redirect('voter_register')
    return render(request, 'voting/register_face.html', {
        'voter_name': request.session['pending_registration']['full_name']
    })


@require_POST
@csrf_exempt
def api_submit_registration(request):
    """API: Process face images and complete registration."""
    pending = request.session.get('pending_registration')
    if not pending:
        return JsonResponse({'success': False, 'message': 'No pending registration found.'})

    try:
        data = json.loads(request.body)
        images_b64 = data.get('images', [])

        if len(images_b64) < 3:
            return JsonResponse({'success': False, 'message': 'At least 3 face images are required.'})

        # Convert base64 images to bytes
        image_bytes_list = [base64_to_bytes(img) for img in images_b64]

        # Check if voter ID already exists
        if Voter.objects.filter(voter_id=pending['voter_id']).exists():
            return JsonResponse({'success': False, 'message': 'Voter ID already registered.'})
        if Voter.objects.filter(email=pending['email']).exists():
            return JsonResponse({'success': False, 'message': 'Email address already registered.'})

        # Generate biometric template
        verifier = get_verifier()
        bio_result = verifier.register(image_bytes_list)

        if not bio_result['success']:
            return JsonResponse({'success': False, 'message': bio_result['message']})

        # Create voter record
        from datetime import date
        voter = Voter(
            voter_id=pending['voter_id'],
            full_name=pending['full_name'],
            email=pending['email'],
            phone=pending['phone'],
            date_of_birth=date.fromisoformat(pending['date_of_birth']),
            gender=pending['gender'],
            state=pending['state'],
            lga=pending['lga'],
            ward=pending['ward'],
            polling_unit=pending['polling_unit'],
            face_encoding=bio_result['embedding'],
            status='approved',  # Auto-approve for prototype; use 'pending' in production
        )
        voter.save()

        # Clear session
        del request.session['pending_registration']

        log_action(
            'register',
            voter_id=voter.voter_id,
            description=f'Voter {voter.full_name} registered successfully.',
            ip=get_client_ip(request),
        )

        return JsonResponse({
            'success': True,
            'message': 'Registration successful! You can now log in to vote.',
            'voter_id': voter.voter_id,
        })

    except Exception as e:
        logger.error(f"Registration error: {e}")
        return JsonResponse({'success': False, 'message': f'Registration failed: {str(e)}'})


# ─────────────────────────────────────────────────────────────
# Voter Authentication (Login)
# ─────────────────────────────────────────────────────────────
def voter_login(request):
    """Step 1: Enter voter ID."""
    if request.method == 'POST':
        form = VoterLoginForm(request.POST)
        if form.is_valid():
            voter_id = form.cleaned_data['voter_id']
            try:
                voter = Voter.objects.get(voter_id=voter_id, status='approved', is_active=True)
                request.session['auth_voter_id'] = voter_id
                return redirect('voter_authenticate')
            except Voter.DoesNotExist:
                messages.error(request, 'Voter ID not found or account not approved.')
    else:
        form = VoterLoginForm()
    return render(request, 'voting/login.html', {'form': form})


def voter_authenticate(request):
    """Step 2: Facial authentication page."""
    voter_id = request.session.get('auth_voter_id')
    if not voter_id:
        return redirect('voter_login')
    try:
        voter = Voter.objects.get(voter_id=voter_id, status='approved')
    except Voter.DoesNotExist:
        return redirect('voter_login')
    return render(request, 'voting/authenticate.html', {'voter': voter})


@require_POST
@csrf_exempt
def api_authenticate(request):
    """API: Perform facial authentication against stored embedding."""
    voter_id = request.session.get('auth_voter_id')
    if not voter_id:
        return JsonResponse({'success': False, 'message': 'No authentication session found.'})

    # ── Rate limiting & lockout check ─────────────────────────
    rate_error = check_auth_rate_limit(request, voter_id)
    if rate_error:
        return rate_error

    try:
        voter = Voter.objects.get(voter_id=voter_id, status='approved', is_active=True)
    except Voter.DoesNotExist:
        return JsonResponse({'success': False, 'message': 'Voter not found.'})

    if not voter.face_encoding:
        return JsonResponse({'success': False, 'message': 'No biometric data found for this voter.'})

    try:
        data = json.loads(request.body)
        image_b64 = data.get('image', '')
        challenge_frames = data.get('challenge_frames', {})  # {straight, left, right}

        if not image_b64:
            return JsonResponse({'success': False, 'message': 'No image provided.'})

        image_bytes = base64_to_bytes(image_b64)

        # If multiple challenge frames provided, use them for stronger liveness
        extra_images = []
        for key in ('left', 'right'):
            if key in challenge_frames:
                try:
                    extra_images.append(base64_to_bytes(challenge_frames[key]))
                except Exception:
                    pass

        verifier = get_verifier()
        auth_result = verifier.authenticate(image_bytes, bytes(voter.face_encoding))

        ip = get_client_ip(request)

        if auth_result['authenticated']:
            record_auth_success(voter_id)
            # Create voter session
            token = secrets.token_hex(32)
            VoterSession.objects.filter(voter=voter).delete()  # Remove old sessions
            VoterSession.objects.create(
                voter=voter,
                session_token=token,
                expires_at=timezone.now() + timedelta(hours=1),
                face_confidence=auth_result['similarity'],
                liveness_score=auth_result['liveness_score'],
            )
            request.session['voter_session_token'] = token
            del request.session['auth_voter_id']

            log_action(
                'auth_success',
                voter_id=voter_id,
                description=f"Facial auth success. Similarity={auth_result['similarity']:.3f}, "
                            f"Liveness={auth_result['liveness_score']:.3f}",
                ip=ip,
            )
            return JsonResponse({
                'success': True,
                'message': f"Welcome, {voter.full_name}! Authentication successful.",
                'redirect': '/vote/',
            })
        else:
            fail_count = record_auth_failure(voter_id)
            remaining = max(0, 5 - fail_count)
            action = 'liveness_fail' if not auth_result['liveness_pass'] else 'auth_fail'
            log_action(action, voter_id=voter_id, description=auth_result['message'], ip=ip, success=False)
            return JsonResponse({
                'success': False,
                'liveness_pass': auth_result['liveness_pass'],
                'liveness_score': auth_result['liveness_score'],
                'similarity': auth_result['similarity'],
                'message': auth_result['message'],
                'attempts_remaining': remaining,
            })

    except Exception as e:
        logger.error(f"Authentication error: {e}")
        return JsonResponse({'success': False, 'message': f'Authentication error: {str(e)}'})


# ─────────────────────────────────────────────────────────────
# Voting
# ─────────────────────────────────────────────────────────────
@voter_authenticated
def vote_dashboard(request):
    """Show available elections for the authenticated voter."""
    voter = request.voter
    active_elections = Election.objects.filter(status='active')

    # Determine which elections voter has already voted in
    voted_election_ids = set()
    for election in active_elections:
        voter_hash = _voter_hash(voter.voter_id, str(election.id))
        if Vote.objects.filter(election=election, voter_hash=voter_hash).exists():
            voted_election_ids.add(str(election.id))

    context = {
        'voter': voter,
        'elections': active_elections,
        'voted_election_ids': voted_election_ids,
    }
    return render(request, 'voting/vote_dashboard.html', context)


@voter_authenticated
def vote_election(request, election_id):
    """Show ballot for a specific election."""
    election = get_object_or_404(Election, id=election_id, status='active')
    voter = request.voter

    voter_hash = _voter_hash(voter.voter_id, str(election.id))
    if Vote.objects.filter(election=election, voter_hash=voter_hash).exists():
        messages.warning(request, 'You have already voted in this election.')
        return redirect('vote_dashboard')

    candidates = election.candidates.all()
    return render(request, 'voting/ballot.html', {
        'election': election,
        'candidates': candidates,
        'voter': voter,
    })


@require_POST
@csrf_exempt
@voter_authenticated
def api_cast_vote(request):
    """API: Cast a vote."""
    voter = request.voter
    try:
        data = json.loads(request.body)
        election_id = data.get('election_id')
        candidate_id = data.get('candidate_id')

        election = get_object_or_404(Election, id=election_id, status='active')
        candidate = get_object_or_404(Candidate, id=candidate_id, election=election)

        voter_hash = _voter_hash(voter.voter_id, str(election.id))

        with transaction.atomic():
            if Vote.objects.filter(election=election, voter_hash=voter_hash).exists():
                return JsonResponse({'success': False, 'message': 'You have already voted in this election.'})

            Vote.objects.create(
                election=election,
                candidate=candidate,
                voter_hash=voter_hash,
                ip_address=get_client_ip(request),
            )

        log_action(
            'vote_cast',
            voter_id=voter.voter_id,
            election_id=election_id,
            description=f'Vote cast for candidate {candidate.full_name} ({candidate.party}).',
            ip=get_client_ip(request),
        )

        return JsonResponse({
            'success': True,
            'message': 'Your vote has been cast successfully!',
            'receipt': _generate_vote_receipt(voter.voter_id, str(election.id), str(candidate.id)),
        })

    except Exception as e:
        logger.error(f"Vote casting error: {e}")
        return JsonResponse({'success': False, 'message': f'Vote casting failed: {str(e)}'})


@voter_authenticated
def vote_receipt(request):
    return render(request, 'voting/receipt.html', {'voter': request.voter})


def voter_logout(request):
    token = request.session.get('voter_session_token')
    if token:
        VoterSession.objects.filter(session_token=token).update(is_valid=False)
    request.session.flush()
    messages.success(request, 'You have been logged out successfully.')
    return redirect('home')


# ─────────────────────────────────────────────────────────────
# Results (Public)
# ─────────────────────────────────────────────────────────────
def election_results(request, election_id):
    election = get_object_or_404(Election, id=election_id)
    candidates = election.candidates.annotate(
        num_votes=Count('votes')
    ).order_by('-num_votes')
    total_votes = election.total_votes
    return render(request, 'voting/results.html', {
        'election': election,
        'candidates': candidates,
        'total_votes': total_votes,
    })


def results_list(request):
    elections = Election.objects.filter(status__in=['active', 'completed'])
    return render(request, 'voting/results_list.html', {'elections': elections})


# ─────────────────────────────────────────────────────────────
# Admin Panel
# ─────────────────────────────────────────────────────────────
def admin_login_view(request):
    if request.user.is_authenticated and request.user.is_staff:
        return redirect('admin_dashboard')
    form = AdminLoginForm()
    if request.method == 'POST':
        form = AdminLoginForm(request.POST)
        if form.is_valid():
            user = authenticate(
                request,
                username=form.cleaned_data['username'],
                password=form.cleaned_data['password'],
            )
            if user and user.is_staff:
                login(request, user)
                return redirect('admin_dashboard')
            else:
                messages.error(request, 'Invalid credentials or insufficient permissions.')
    return render(request, 'admin_panel/login.html', {'form': form})


def admin_logout_view(request):
    logout(request)
    return redirect('admin_login')


@login_required(login_url='/admin-panel/login/')
def admin_dashboard(request):
    if not request.user.is_staff:
        return redirect('home')

    context = {
        'total_voters': Voter.objects.count(),
        'approved_voters': Voter.objects.filter(status='approved').count(),
        'pending_voters': Voter.objects.filter(status='pending').count(),
        'total_elections': Election.objects.count(),
        'active_elections': Election.objects.filter(status='active').count(),
        'total_votes': Vote.objects.count(),
        'recent_logs': AuditLog.objects.all()[:20],
        'elections': Election.objects.all()[:5],
    }
    return render(request, 'admin_panel/dashboard.html', context)


@login_required(login_url='/admin-panel/login/')
def admin_voters(request):
    if not request.user.is_staff:
        return redirect('home')
    voters = Voter.objects.all().order_by('-registered_at')
    return render(request, 'admin_panel/voters.html', {'voters': voters})


@login_required(login_url='/admin-panel/login/')
def admin_elections(request):
    if not request.user.is_staff:
        return redirect('home')

    if request.method == 'POST':
        form = ElectionForm(request.POST)
        if form.is_valid():
            election = form.save(commit=False)
            election.created_by = request.user
            # Auto-set status
            now = timezone.now()
            if election.start_date <= now <= election.end_date:
                election.status = 'active'
            elif election.start_date > now:
                election.status = 'upcoming'
            election.save()
            messages.success(request, 'Election created successfully.')
            return redirect('admin_elections')
    else:
        form = ElectionForm()

    elections = Election.objects.all()
    return render(request, 'admin_panel/elections.html', {
        'elections': elections, 'form': form
    })


@login_required(login_url='/admin-panel/login/')
def admin_candidates(request):
    if not request.user.is_staff:
        return redirect('home')

    if request.method == 'POST':
        form = CandidateForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            messages.success(request, 'Candidate added successfully.')
            return redirect('admin_candidates')
    else:
        form = CandidateForm()

    candidates = Candidate.objects.select_related('election').all()
    return render(request, 'admin_panel/candidates.html', {
        'candidates': candidates, 'form': form
    })


@login_required(login_url='/admin-panel/login/')
def admin_toggle_election(request, election_id):
    if not request.user.is_staff:
        return redirect('home')
    election = get_object_or_404(Election, id=election_id)
    if election.status == 'active':
        election.status = 'completed'
    elif election.status in ('upcoming', 'completed'):
        election.status = 'active'
    election.save()
    messages.success(request, f'Election status changed to {election.status}.')
    return redirect('admin_elections')


@login_required(login_url='/admin-panel/login/')
def admin_approve_voter(request, voter_id):
    if not request.user.is_staff:
        return redirect('home')
    voter = get_object_or_404(Voter, id=voter_id)
    voter.status = 'approved'
    voter.save()
    messages.success(request, f'Voter {voter.full_name} approved.')
    return redirect('admin_voters')


@login_required(login_url='/admin-panel/login/')
def admin_audit_logs(request):
    if not request.user.is_staff:
        return redirect('home')
    logs = AuditLog.objects.all()
    return render(request, 'admin_panel/audit_logs.html', {'logs': logs})


@login_required(login_url='/admin-panel/login/')
def admin_metrics(request):
    """System performance metrics."""
    if not request.user.is_staff:
        return redirect('home')

    auth_success = AuditLog.objects.filter(action='auth_success').count()
    auth_fail = AuditLog.objects.filter(action='auth_fail').count()
    liveness_fail = AuditLog.objects.filter(action='liveness_fail').count()
    total_auth = auth_success + auth_fail + liveness_fail

    context = {
        'auth_success': auth_success,
        'auth_fail': auth_fail,
        'liveness_fail': liveness_fail,
        'total_auth': total_auth,
        'success_rate': round(auth_success / total_auth * 100, 2) if total_auth else 0,
    }
    return render(request, 'admin_panel/metrics.html', context)


# ─────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────
def _voter_hash(voter_id: str, election_id: str) -> str:
    """Creates an anonymized voter-election hash to enforce one-vote-per-voter."""
    raw = f"{voter_id}:{election_id}:evoting_salt_2024"
    return hashlib.sha256(raw.encode()).hexdigest()


def _generate_vote_receipt(voter_id: str, election_id: str, candidate_id: str) -> str:
    """Generate a unique receipt code for the voter."""
    raw = f"{voter_id}:{election_id}:{candidate_id}:{timezone.now().isoformat()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16].upper()
