#!/usr/bin/env python
"""
setup.py — One-time setup script for NigeriaVotes E-Voting System
Run: python setup.py
"""
import os
import sys
import subprocess

def run(cmd, check=True):
    print(f"  $ {cmd}")
    result = subprocess.run(cmd, shell=True, check=check)
    return result

def main():
    print("\n" + "="*60)
    print("  NigeriaVotes E-Voting System — Setup")
    print("="*60)

    # 1. Run migrations
    print("\n[1/4] Running database migrations...")
    run("python manage.py migrate")

    # 2. Create superuser
    print("\n[2/4] Creating admin superuser...")
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'evoting.settings')
    import django
    django.setup()

    from django.contrib.auth.models import User
    if not User.objects.filter(username='admin').exists():
        User.objects.create_superuser(
            username='admin',
            email='admin@nigerialvotes.gov.ng',
            password='Admin@2025!'
        )
        print("  ✅ Admin user created: username=admin  password=Admin@2025!")
    else:
        print("  ℹ️  Admin user already exists.")

    # 3. Collect static files
    print("\n[3/4] Collecting static files...")
    run("python manage.py collectstatic --noinput", check=False)

    # 4. Create sample election (optional)
    print("\n[4/4] Creating sample election data...")
    from django.utils import timezone
    from datetime import timedelta
    from voting.models import Election, Candidate

    if not Election.objects.exists():
        election = Election.objects.create(
            title="2027 Presidential Election",
            election_type="presidential",
            description="Nigeria's 2027 general presidential election — prototype demonstration",
            start_date=timezone.now() - timedelta(hours=1),  # already started
            end_date=timezone.now() + timedelta(days=7),
            status='active',
            created_by=User.objects.filter(is_superuser=True).first(),
        )

        candidates_data = [
            {"full_name": "Aisha Bello", "party": "APC", "party_name": "All Progressives Congress", "position_number": 1, "bio": "Former governor with 12 years of public service experience."},
            {"full_name": "Emeka Okonkwo", "party": "PDP", "party_name": "Peoples Democratic Party", "position_number": 2, "bio": "Economist and former Minister of Finance."},
            {"full_name": "Fatima Al-Hassan", "party": "LP", "party_name": "Labour Party", "position_number": 3, "bio": "Human rights lawyer and civil society advocate."},
            {"full_name": "Tunde Adeyemi", "party": "NNPP", "party_name": "New Nigeria Peoples Party", "position_number": 4, "bio": "Business leader and philanthropist from Ogun State."},
        ]

        for cdata in candidates_data:
            Candidate.objects.create(election=election, **cdata)

        print(f"  ✅ Sample election created: '{election.title}' with {len(candidates_data)} candidates.")
    else:
        print("  ℹ️  Sample data already exists.")

    print("\n" + "="*60)
    print("  ✅ Setup complete!")
    print("="*60)
    print()
    print("  Start the server:   python manage.py runserver")
    print("  Public site:        http://127.0.0.1:8000/")
    print("  Admin panel:        http://127.0.0.1:8000/admin-panel/")
    print("  Django admin:       http://127.0.0.1:8000/admin/")
    print()
    print("  Admin credentials:  username=admin  password=Admin@2025!")
    print("="*60 + "\n")

if __name__ == '__main__':
    main()
