from django.urls import path
from . import views

urlpatterns = [
    # Public
    path('', views.home, name='home'),
    path('results/', views.results_list, name='results_list'),
    path('results/<uuid:election_id>/', views.election_results, name='election_results'),

    # Voter Registration
    path('register/', views.voter_register, name='voter_register'),
    path('register/face/', views.voter_register_face, name='voter_register_face'),

    # Voter Authentication
    path('login/', views.voter_login, name='voter_login'),
    path('authenticate/', views.voter_authenticate, name='voter_authenticate'),
    path('logout/', views.voter_logout, name='voter_logout'),

    # Voting
    path('vote/', views.vote_dashboard, name='vote_dashboard'),
    path('vote/<uuid:election_id>/', views.vote_election, name='vote_election'),
    path('vote/receipt/', views.vote_receipt, name='vote_receipt'),

    # APIs (AJAX endpoints)
    path('api/register/submit/', views.api_submit_registration, name='api_submit_registration'),
    path('api/authenticate/', views.api_authenticate, name='api_authenticate'),
    path('api/vote/cast/', views.api_cast_vote, name='api_cast_vote'),

    # Admin Panel
    path('admin-panel/login/', views.admin_login_view, name='admin_login'),
    path('admin-panel/logout/', views.admin_logout_view, name='admin_logout'),
    path('admin-panel/', views.admin_dashboard, name='admin_dashboard'),
    path('admin-panel/voters/', views.admin_voters, name='admin_voters'),
    path('admin-panel/voters/<uuid:voter_id>/approve/', views.admin_approve_voter, name='admin_approve_voter'),
    path('admin-panel/elections/', views.admin_elections, name='admin_elections'),
    path('admin-panel/elections/<uuid:election_id>/toggle/', views.admin_toggle_election, name='admin_toggle_election'),
    path('admin-panel/candidates/', views.admin_candidates, name='admin_candidates'),
    path('admin-panel/audit/', views.admin_audit_logs, name='admin_audit_logs'),
    path('admin-panel/metrics/', views.admin_metrics, name='admin_metrics'),
]
