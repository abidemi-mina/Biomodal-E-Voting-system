from django.contrib import admin
from .models import Voter, Election, Candidate, Vote, AuditLog, VoterSession


@admin.register(Voter)
class VoterAdmin(admin.ModelAdmin):
    list_display = ['voter_id', 'full_name', 'email', 'state', 'status', 'registered_at']
    list_filter = ['status', 'state', 'gender']
    search_fields = ['voter_id', 'full_name', 'email']
    readonly_fields = ['id', 'registered_at', 'face_encoding']
    list_per_page = 50


@admin.register(Election)
class ElectionAdmin(admin.ModelAdmin):
    list_display = ['title', 'election_type', 'status', 'start_date', 'end_date', 'total_votes']
    list_filter = ['status', 'election_type']
    search_fields = ['title']
    readonly_fields = ['id', 'created_at']


@admin.register(Candidate)
class CandidateAdmin(admin.ModelAdmin):
    list_display = ['full_name', 'party', 'election', 'position_number', 'vote_count']
    list_filter = ['party', 'election']
    search_fields = ['full_name']


@admin.register(Vote)
class VoteAdmin(admin.ModelAdmin):
    list_display = ['election', 'candidate', 'voted_at', 'ip_address']
    list_filter = ['election', 'candidate__party']
    readonly_fields = ['id', 'voter_hash', 'voted_at']


@admin.register(AuditLog)
class AuditLogAdmin(admin.ModelAdmin):
    list_display = ['action', 'voter_id', 'success', 'timestamp', 'ip_address']
    list_filter = ['action', 'success']
    search_fields = ['voter_id', 'description']
    readonly_fields = ['id', 'timestamp']


@admin.register(VoterSession)
class VoterSessionAdmin(admin.ModelAdmin):
    list_display = ['voter', 'authenticated_at', 'expires_at', 'face_confidence', 'liveness_score', 'is_valid']
    list_filter = ['is_valid']
    readonly_fields = ['session_token']
