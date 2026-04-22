"""
setup_demo.py
=============
Populates the database with sample elections and candidates for demo/testing.

Usage:
    python manage.py shell < setup_demo.py
    OR
    python setup_demo.py
"""
import os
import sys
import django
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'evoting.settings')
django.setup()

from django.contrib.auth.models import User
from django.utils import timezone
from datetime import timedelta
from voting.models import Election, Candidate, AuditLog


def create_superuser():
    if not User.objects.filter(username='admin').exists():
        User.objects.create_superuser('admin', 'admin@nigeria.gov.ng', 'admin123')
        print("✅ Superuser created: admin / admin123")
    else:
        print("ℹ️  Superuser already exists.")


def create_elections():
    now = timezone.now()

    # Presidential election
    pres, created = Election.objects.get_or_create(
        title='2027 Presidential Election',
        defaults={
            'election_type': 'presidential',
            'description': 'The general presidential election to choose the next President of the Federal Republic of Nigeria.',
            'start_date': now - timedelta(hours=2),
            'end_date': now + timedelta(hours=22),
            'status': 'active',
        }
    )
    if created:
        print(f"✅ Created election: {pres.title}")
        Candidate.objects.create(election=pres, full_name='Adeola Chukwuemeka Johnson', party='APC', position_number=1,
            bio='Former Governor of Lagos State. 20 years in public service. Champion of digital governance.')
        Candidate.objects.create(election=pres, full_name='Fatima Bello Idris', party='PDP', position_number=2,
            bio='Economist, former Finance Minister. 3 terms as Senator from Kano State.')
        Candidate.objects.create(election=pres, full_name='Emeka Okafor Nwosu', party='LP', position_number=3,
            bio='Civil rights lawyer and youth activist. First-time presidential candidate.')
        Candidate.objects.create(election=pres, full_name='Auwal Musa Barau', party='NNPP', position_number=4,
            bio='Businessman and philanthropist from Katsina.')
        print(f"  ✅ Added 4 presidential candidates")

    # Governorship election
    gov, created = Election.objects.get_or_create(
        title='2027 Oyo State Gubernatorial Election',
        defaults={
            'election_type': 'gubernatorial',
            'description': 'Election for the Governor of Oyo State.',
            'start_date': now - timedelta(hours=1),
            'end_date': now + timedelta(hours=23),
            'status': 'active',
        }
    )
    if created:
        print(f"✅ Created election: {gov.title}")
        Candidate.objects.create(election=gov, full_name='Dr. Babatunde Adegboyega', party='APC', position_number=1,
            bio='Professor of Medicine, University of Ibadan. Passionate about healthcare reform.')
        Candidate.objects.create(election=gov, full_name='Hajia Ramatu Lawal', party='PDP', position_number=2,
            bio='Current Deputy Governor. Master of Public Administration, Harvard University.')
        Candidate.objects.create(election=gov, full_name='Oluwaseun Abiodun', party='LP', position_number=3,
            bio='Tech entrepreneur and social justice advocate.')
        print(f"  ✅ Added 3 gubernatorial candidates")

    # Senate election
    sen, created = Election.objects.get_or_create(
        title='2027 Lagos Central Senatorial Election',
        defaults={
            'election_type': 'senate',
            'description': 'Election for the Senator representing Lagos Central District.',
            'start_date': now + timedelta(days=7),
            'end_date': now + timedelta(days=7, hours=24),
            'status': 'upcoming',
        }
    )
    if created:
        print(f"✅ Created election: {sen.title}")
        Candidate.objects.create(election=sen, full_name='Olawale Adeyemi', party='APC', position_number=1,
            bio='Current Senator, seeking second term.')
        Candidate.objects.create(election=sen, full_name='Chidinma Okonkwo', party='PDP', position_number=2,
            bio='Lawyer and women\'s rights activist.')
        print(f"  ✅ Added 2 senate candidates")

    return pres, gov, sen


def main():
    print("\n🗳️  NigeriaVotes Demo Setup")
    print("=" * 50)
    create_superuser()
    pres, gov, sen = create_elections()
    print("\n✅ Setup complete!")
    print("\nAccess points:")
    print("  Public site:    http://127.0.0.1:8000/")
    print("  Admin panel:    http://127.0.0.1:8000/admin-panel/")
    print("  Django admin:   http://127.0.0.1:8000/admin/")
    print("\nAdmin credentials: admin / admin123")
    print("\nActive elections:")
    print(f"  • {pres.title} (Presidential)")
    print(f"  • {gov.title} (Gubernatorial)")
    print(f"  • {sen.title} (Upcoming - Senate)")


if __name__ == '__main__':
    main()
