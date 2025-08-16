import re
import streamlit as st
from datetime import datetime, timedelta
from pydantic import BaseModel, Field, validator, EmailStr
from typing import Optional, Dict, Any
import time


class UserContact(BaseModel):
    """User contact information with validation"""
    name: str = Field(..., min_length=2, max_length=100)
    phone: str = Field(..., min_length=10, max_length=15)
    email: EmailStr

    @validator('name')
    def validate_name(cls, v):
        if not re.match(r'^[a-zA-Z\s]+$', v):
            raise ValueError('Name should only contain letters and spaces')
        return v.strip().title()

    @validator('phone')
    def validate_phone(cls, v):
        # Remove all non-digit characters
        phone_digits = re.sub(r'\D', '', v)
        if len(phone_digits) < 10 or len(phone_digits) > 15:
            raise ValueError('Phone number should be between 10-15 digits')
        return phone_digits


class AppointmentBooking(BaseModel):
    """Appointment booking with validation"""
    user_info: UserContact
    appointment_date: str = Field(..., pattern=r'^\d{4}-\d{2}-\d{2}$')
    appointment_time: str = Field(..., pattern=r'^\d{2}:\d{2}$')
    purpose: str = Field(..., min_length=5, max_length=500)

    @validator('appointment_date')
    def validate_date(cls, v):
        try:
            date_obj = datetime.strptime(v, '%Y-%m-%d')
            if date_obj.date() < datetime.now().date():
                raise ValueError('Appointment date cannot be in the past')
            return v
        except ValueError as e:
            if 'does not match format' in str(e):
                raise ValueError('Date must be in YYYY-MM-DD format')
            raise e


class ConversationalForm:
    """Handles conversational form collection with validation"""

    def __init__(self, gemini_chat):
        self.gemini_chat = gemini_chat
        self.current_form = None
        self.form_data = {}
        self.form_step = 0
        self.form_fields = []

    def detect_booking_intent(self, user_message):
        """Detect if user wants to book an appointment or callback"""
        booking_keywords = [
            'call me', 'book appointment', 'schedule', 'appointment',
            'meeting', 'callback', 'call back', 'contact me', 'book a call'
        ]

        message_lower = user_message.lower()
        return any(keyword in message_lower for keyword in booking_keywords)

    def start_contact_form(self):
        """Start collecting contact information"""
        self.current_form = "contact"
        self.form_data = {}
        self.form_step = 0
        self.form_fields = ['name', 'phone', 'email']

        return "I'd be happy to arrange a callback for you! Let me collect some information.\n\n**What's your full name?**"

    def start_appointment_form(self):
        """Start collecting appointment information"""
        self.current_form = "appointment"
        self.form_data = {}
        self.form_step = 0
        self.form_fields = ['name', 'phone',
                            'email', 'date', 'time', 'purpose']

        return "I'll help you book an appointment! Let me collect some details.\n\n**What's your full name?**"

    def process_form_input(self, user_input):
        """Process user input for the current form step"""
        if not self.current_form:
            return None

        current_field = self.form_fields[self.form_step]

        try:
            if current_field == 'name':
                return self._process_name(user_input)
            elif current_field == 'phone':
                return self._process_phone(user_input)
            elif current_field == 'email':
                return self._process_email(user_input)
            elif current_field == 'date':
                return self._process_date(user_input)
            elif current_field == 'time':
                return self._process_time(user_input)
            elif current_field == 'purpose':
                return self._process_purpose(user_input)

        except Exception as e:
            return f"❌ {str(e)}\n\nPlease try again."

    def _process_name(self, name):
        """Process and validate name input"""
        try:
            validated_name = UserContact(
                name=name, phone="1234567890", email="test@test.com").name
            self.form_data['name'] = validated_name
            self.form_step += 1

            return f"Thanks {validated_name}! 📱 **What's your phone number?**"

        except Exception as e:
            raise ValueError(
                "Please enter a valid name (letters and spaces only)")

    def _process_phone(self, phone):
        """Process and validate phone input"""
        try:
            validated_phone = UserContact(
                name="Test", phone=phone, email="test@test.com").phone
            self.form_data['phone'] = validated_phone
            self.form_step += 1

            return f"Got it! 📧 **What's your email address?**"

        except Exception as e:
            raise ValueError(
                "Please enter a valid phone number (10-15 digits)")

    def _process_email(self, email):
        """Process and validate email input"""
        try:
            UserContact(name="Test", phone="1234567890", email=email)
            self.form_data['email'] = email
            self.form_step += 1

            if self.current_form == "contact":
                return self._complete_contact_form()
            else:
                return f"Perfect! 📅 **When would you like to schedule the appointment?** (e.g., 'next Monday', '2024-12-25', 'tomorrow')"

        except Exception as e:
            raise ValueError("Please enter a valid email address")

    def _process_date(self, date_input):
        """Process and validate date input with natural language parsing"""
        try:
            parsed_date = self._parse_natural_date(date_input)
            self.form_data['date'] = parsed_date
            self.form_step += 1

            return f"Great! Date set for {parsed_date}. 🕐 **What time would you prefer?** (e.g., '10:30', '2:00 PM', '14:00')"

        except Exception as e:
            raise ValueError(f"Please enter a valid date. {str(e)}")

    def _process_time(self, time_input):
        """Process and validate time input"""
        try:
            parsed_time = self._parse_time(time_input)
            self.form_data['time'] = parsed_time
            self.form_step += 1

            return f"Time set for {parsed_time}. 📝 **What's the purpose of this appointment?** (brief description)"

        except Exception as e:
            raise ValueError(
                "Please enter a valid time (e.g., '10:30', '2:00 PM', '14:00')")

    def _process_purpose(self, purpose):
        """Process and validate appointment purpose"""
        if len(purpose.strip()) < 5:
            raise ValueError(
                "Please provide a more detailed purpose (at least 5 characters)")

        self.form_data['purpose'] = purpose.strip()
        return self._complete_appointment_form()

    def _parse_natural_date(self, date_input):
        """Parse natural language date input"""
        date_input = date_input.lower().strip()
        today = datetime.now().date()

        # Handle relative dates
        if 'today' in date_input:
            return today.strftime('%Y-%m-%d')
        elif 'tomorrow' in date_input:
            return (today + timedelta(days=1)).strftime('%Y-%m-%d')
        elif 'next monday' in date_input:
            days_ahead = 0 - today.weekday()  # Monday is 0
            if days_ahead <= 0:  # Target day already happened this week
                days_ahead += 7
            return (today + timedelta(days=days_ahead)).strftime('%Y-%m-%d')
        elif 'next tuesday' in date_input:
            days_ahead = 1 - today.weekday()
            if days_ahead <= 0:
                days_ahead += 7
            return (today + timedelta(days=days_ahead)).strftime('%Y-%m-%d')
        elif 'next wednesday' in date_input:
            days_ahead = 2 - today.weekday()
            if days_ahead <= 0:
                days_ahead += 7
            return (today + timedelta(days=days_ahead)).strftime('%Y-%m-%d')
        elif 'next thursday' in date_input:
            days_ahead = 3 - today.weekday()
            if days_ahead <= 0:
                days_ahead += 7
            return (today + timedelta(days=days_ahead)).strftime('%Y-%m-%d')
        elif 'next friday' in date_input:
            days_ahead = 4 - today.weekday()
            if days_ahead <= 0:
                days_ahead += 7
            return (today + timedelta(days=days_ahead)).strftime('%Y-%m-%d')

        # Try to parse YYYY-MM-DD format
        try:
            if re.match(r'^\d{4}-\d{2}-\d{2}$', date_input):
                parsed_date = datetime.strptime(date_input, '%Y-%m-%d').date()
                if parsed_date < today:
                    raise ValueError("Date cannot be in the past")
                return date_input
        except:
            pass

        # Try other common formats
        common_formats = ['%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%d-%m-%Y']
        for fmt in common_formats:
            try:
                parsed_date = datetime.strptime(date_input, fmt).date()
                if parsed_date < today:
                    raise ValueError("Date cannot be in the past")
                return parsed_date.strftime('%Y-%m-%d')
            except:
                continue

        raise ValueError(
            "Please use a format like 'YYYY-MM-DD', 'tomorrow', or 'next Monday'")

    def _parse_time(self, time_input):
        """Parse time input in various formats"""
        time_input = time_input.lower().strip()

        # Handle 12-hour format
        if 'am' in time_input or 'pm' in time_input:
            try:
                time_obj = datetime.strptime(
                    time_input.replace(' ', ''), '%I:%M%p')
                return time_obj.strftime('%H:%M')
            except:
                try:
                    time_obj = datetime.strptime(
                        time_input.replace(' ', ''), '%I%p')
                    return time_obj.strftime('%H:%M')
                except:
                    pass

        # Handle 24-hour format
        try:
            time_obj = datetime.strptime(time_input, '%H:%M')
            return time_obj.strftime('%H:%M')
        except:
            pass

        # Handle hour only
        try:
            hour = int(time_input)
            if 0 <= hour <= 23:
                return f"{hour:02d}:00"
        except:
            pass

        raise ValueError(
            "Please enter time in format like '10:30', '2:00 PM', or '14:00'")

    def _complete_contact_form(self):
        """Complete contact form and trigger callback booking"""
        try:
            contact_info = UserContact(
                name=self.form_data['name'],
                phone=self.form_data['phone'],
                email=self.form_data['email']
            )

            # Reset form
            self.current_form = None
            self.form_data = {}
            self.form_step = 0

            # Trigger booking process
            return self._book_callback(contact_info)

        except Exception as e:
            return f"❌ Error completing form: {str(e)}"

    def _complete_appointment_form(self):
        """Complete appointment form and trigger booking"""
        try:
            contact_info = UserContact(
                name=self.form_data['name'],
                phone=self.form_data['phone'],
                email=self.form_data['email']
            )

            appointment = AppointmentBooking(
                user_info=contact_info,
                appointment_date=self.form_data['date'],
                appointment_time=self.form_data['time'],
                purpose=self.form_data['purpose']
            )

            # Reset form
            self.current_form = None
            self.form_data = {}
            self.form_step = 0

            # Trigger booking process
            return self._book_appointment(appointment)

        except Exception as e:
            return f"❌ Error completing form: {str(e)}"

    def _book_callback(self, contact_info):
        """Simulate callback booking process"""
        return f"""✅ **Callback Request Submitted!**

📋 **Your Details:**
- **Name:** {contact_info.name}
- **Phone:** {contact_info.phone}
- **Email:** {contact_info.email}

🔄 **Processing booking...**"""

    def _book_appointment(self, appointment):
        """Simulate appointment booking process"""
        return f"""✅ **Appointment Booking Submitted!**

📋 **Your Details:**
- **Name:** {appointment.user_info.name}
- **Phone:** {appointment.user_info.phone}
- **Email:** {appointment.user_info.email}

📅 **Appointment Details:**
- **Date:** {appointment.appointment_date}
- **Time:** {appointment.appointment_time}
- **Purpose:** {appointment.purpose}

🔄 **Processing booking...**"""

    def is_form_active(self):
        """Check if a form is currently active"""
        return self.current_form is not None

    def cancel_form(self):
        """Cancel the current form"""
        self.current_form = None
        self.form_data = {}
        self.form_step = 0
        return "❌ Form cancelled. How else can I help you?"


def simulate_booking_process():
    """Simulate the booking process with delay"""
    time.sleep(10)
    return "🎉 **Booking Confirmed!** You will receive a confirmation email shortly."
