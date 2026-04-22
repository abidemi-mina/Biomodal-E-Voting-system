from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from .models import Voter, Election, Candidate
from django.utils import timezone


NIGERIAN_STATES = [
    ('', '-- Select State --'),
    ('Abia', 'Abia'), ('Adamawa', 'Adamawa'), ('Akwa Ibom', 'Akwa Ibom'),
    ('Anambra', 'Anambra'), ('Bauchi', 'Bauchi'), ('Bayelsa', 'Bayelsa'),
    ('Benue', 'Benue'), ('Borno', 'Borno'), ('Cross River', 'Cross River'),
    ('Delta', 'Delta'), ('Ebonyi', 'Ebonyi'), ('Edo', 'Edo'),
    ('Ekiti', 'Ekiti'), ('Enugu', 'Enugu'), ('FCT', 'FCT - Abuja'),
    ('Gombe', 'Gombe'), ('Imo', 'Imo'), ('Jigawa', 'Jigawa'),
    ('Kaduna', 'Kaduna'), ('Kano', 'Kano'), ('Katsina', 'Katsina'),
    ('Kebbi', 'Kebbi'), ('Kogi', 'Kogi'), ('Kwara', 'Kwara'),
    ('Lagos', 'Lagos'), ('Nasarawa', 'Nasarawa'), ('Niger', 'Niger'),
    ('Ogun', 'Ogun'), ('Ondo', 'Ondo'), ('Osun', 'Osun'),
    ('Oyo', 'Oyo'), ('Plateau', 'Plateau'), ('Rivers', 'Rivers'),
    ('Sokoto', 'Sokoto'), ('Taraba', 'Taraba'), ('Yobe', 'Yobe'),
    ('Zamfara', 'Zamfara'),
]


class VoterRegistrationForm(forms.ModelForm):
    state = forms.ChoiceField(choices=NIGERIAN_STATES)
    date_of_birth = forms.DateField(
        widget=forms.DateInput(attrs={'type': 'date'}),
        help_text='You must be at least 18 years old.'
    )
    confirm_email = forms.EmailField(label='Confirm Email')

    class Meta:
        model = Voter
        fields = [
            'voter_id', 'full_name', 'email', 'phone',
            'date_of_birth', 'gender', 'state', 'lga',
            'ward', 'polling_unit',
        ]
        widgets = {
            'voter_id': forms.TextInput(attrs={'placeholder': 'e.g. 00AB123456789'}),
            'full_name': forms.TextInput(attrs={'placeholder': 'Full legal name'}),
            'phone': forms.TextInput(attrs={'placeholder': '+234XXXXXXXXXX'}),
            'lga': forms.TextInput(attrs={'placeholder': 'Local Government Area'}),
            'ward': forms.TextInput(attrs={'placeholder': 'Electoral Ward'}),
            'polling_unit': forms.TextInput(attrs={'placeholder': 'Polling Unit Name/Code'}),
        }

    def clean(self):
        cleaned_data = super().clean()
        email = cleaned_data.get('email')
        confirm_email = cleaned_data.get('confirm_email')
        if email and confirm_email and email != confirm_email:
            raise forms.ValidationError('Email addresses do not match.')

        dob = cleaned_data.get('date_of_birth')
        if dob:
            from datetime import date
            age = (date.today() - dob).days // 365
            if age < 18:
                raise forms.ValidationError('Voter must be at least 18 years old.')
        return cleaned_data


class VoterLoginForm(forms.Form):
    voter_id = forms.CharField(
        max_length=20,
        label='Voter ID',
        widget=forms.TextInput(attrs={'placeholder': 'Enter your Voter ID'})
    )


class ElectionForm(forms.ModelForm):
    class Meta:
        model = Election
        fields = ['title', 'election_type', 'description', 'start_date', 'end_date']
        widgets = {
            'start_date': forms.DateTimeInput(attrs={'type': 'datetime-local'}),
            'end_date': forms.DateTimeInput(attrs={'type': 'datetime-local'}),
            'description': forms.Textarea(attrs={'rows': 3}),
        }

    def clean(self):
        cleaned_data = super().clean()
        start = cleaned_data.get('start_date')
        end = cleaned_data.get('end_date')
        if start and end and end <= start:
            raise forms.ValidationError('End date must be after start date.')
        return cleaned_data


class CandidateForm(forms.ModelForm):
    class Meta:
        model = Candidate
        fields = ['election', 'full_name', 'party', 'party_name', 'bio', 'photo', 'position_number']
        widgets = {
            'bio': forms.Textarea(attrs={'rows': 3}),
        }


class AdminLoginForm(forms.Form):
    username = forms.CharField(widget=forms.TextInput(attrs={'placeholder': 'Admin username'}))
    password = forms.CharField(widget=forms.PasswordInput(attrs={'placeholder': 'Password'}))
