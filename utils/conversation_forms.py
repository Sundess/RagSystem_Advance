import re
from datetime import datetime, timedelta
from pydantic import BaseModel, Field, field_validator, EmailStr
from typing import Optional


class ContactInfo(BaseModel):
    """Validated contact information"""
    name: str = Field(..., min_length=2, max_length=100)
    phone: str = Field(..., min_length=10, max_length=10)
    email: EmailStr

    @field_validator('name')
    def validate_name(cls, v):
        if not re.match(r'^[a-zA-Z\s]+$', v):
            raise ValueError('Name should only contain letters and spaces')
        return v.strip().title()

    @field_validator('phone')
    def validate_phone(cls, v):
        phone_digits = re.sub(r'\D', '', v)
        if len(phone_digits) < 10 or len(phone_digits) > 10:
            raise ValueError('Phone number should be 10 digits')
        return phone_digits


class AppointmentDetails(BaseModel):
    """Appointment details with validation"""
    contact: ContactInfo
    date: str = Field(..., pattern=r'^\d{4}-\d{2}-\d{2}$')
    time: str = Field(..., pattern=r'^\d{2}:\d{2}$')
    purpose: str = Field(..., min_length=5, max_length=500)

    @field_validator('date')
    def validate_date(cls, v):
        try:
            date_obj = datetime.strptime(v, '%Y-%m-%d')
            if date_obj.date() < datetime.now().date():
                raise ValueError('Date cannot be in the past')
            return v
        except ValueError as e:
            if 'does not match format' in str(e):
                raise ValueError('Use YYYY-MM-DD format')
            raise e


class ConversationalForm:
    """Simplified conversational form handler"""

    def __init__(self, gemini_chat):
        self.gemini_chat = gemini_chat
        self.reset_form()

    def reset_form(self):
        """Reset form state"""
        self.form_type = None  # 'callback' or 'appointment'
        self.current_step = 0
        self.data = {}
        self.steps = []

    def detect_booking_intent(self, message):
        """Detect booking intent in user message"""
        booking_keywords = [
            'call me', 'book appointment', 'schedule', 'appointment',
            'meeting', 'callback', 'call back', 'contact me', 'book a call',
            'book meeting', 'arrange meeting', 'set up meeting', 'book an appointment',
            'schedule meeting', 'schedule appointment', 'arrange appointment',
            'schedule call', 'arrange call', 'book me', 'set up appointment'
        ]
        message_lower = message.lower().strip()
        detected = any(
            keyword in message_lower for keyword in booking_keywords)
        # Debug
        print(f"🔍 Booking intent detection for '{message}': {detected}")
        return detected

    def start_callback_form(self):
        """Start callback booking process"""
        self.form_type = 'callback'
        self.current_step = 0
        self.steps = ['name', 'phone', 'email']
        return "I'll arrange a callback for you! 📞\n\n**What's your full name?**"

    def start_appointment_form(self):
        """Start appointment booking process"""
        self.form_type = 'appointment'
        self.current_step = 0
        self.steps = ['name', 'phone', 'email', 'date', 'time', 'purpose']
        return "Let's book your appointment! 📅\n\n**What's your full name?**"

    def process_step(self, user_input):
        """Process current form step"""
        if not self.is_active():
            return None

        current_field = self.steps[self.current_step]

        try:
            if current_field == 'name':
                return self._handle_name(user_input)
            elif current_field == 'phone':
                return self._handle_phone(user_input)
            elif current_field == 'email':
                return self._handle_email(user_input)
            elif current_field == 'date':
                return self._handle_date(user_input)
            elif current_field == 'time':
                return self._handle_time(user_input)
            elif current_field == 'purpose':
                return self._handle_purpose(user_input)

        except Exception as e:
            return f"❌ {str(e)}\n\nPlease try again."

    def _handle_name(self, name):
        """Handle name input"""
        # Validate name
        test_contact = ContactInfo(
            name=name, phone="1234567890", email="test@test.com")
        self.data['name'] = test_contact.name
        self.current_step += 1
        return f"Thanks {test_contact.name}! 📱\n\n**What's your phone number?**"

    def _handle_phone(self, phone):
        """Handle phone input"""
        test_contact = ContactInfo(
            name="Test", phone=phone, email="test@test.com")
        self.data['phone'] = test_contact.phone
        self.current_step += 1
        return "Great! 📧\n\n**What's your email address?**"

    def _handle_email(self, email):
        """Handle email input"""
        ContactInfo(name="Test", phone="1234567890", email=email)  # Validate
        self.data['email'] = email
        self.current_step += 1

        if self.form_type == 'callback':
            return self._complete_callback()
        else:
            return ("Perfect! 📅\n\n**When would you like the appointment?**\n"
                    "(e.g., '2024-12-25', 'tomorrow', 'next Monday')")

    def _handle_date(self, date_input):
        """Handle date input with natural language parsing"""
        parsed_date = self._parse_date(date_input)
        self.data['date'] = parsed_date
        self.current_step += 1
        return (f"Date set for {parsed_date}! 🕐\n\n"
                "**What time would you prefer?**\n(e.g., '10:30', '2:00 PM', '14:00')")

    def _handle_time(self, time_input):
        """Handle time input"""
        parsed_time = self._parse_time(time_input)
        self.data['time'] = parsed_time
        self.current_step += 1
        return (f"Time set for {parsed_time}! 📝\n\n"
                "**What's the purpose of your appointment?**")

    def _handle_purpose(self, purpose):
        """Handle purpose input"""
        if len(purpose.strip()) < 5:
            raise ValueError(
                "Please provide more details (at least 5 characters)")

        self.data['purpose'] = purpose.strip()
        return self._complete_appointment()

    def _parse_date(self, date_input):
        """Parse date with natural language support"""
        date_input = date_input.lower().strip()
        today = datetime.now().date()

        # Handle relative dates
        if date_input == 'today':
            return today.strftime('%Y-%m-%d')
        elif date_input == 'tomorrow':
            return (today + timedelta(days=1)).strftime('%Y-%m-%d')
        elif 'next monday' in date_input:
            days_ahead = 0 - today.weekday()
            if days_ahead <= 0:
                days_ahead += 7
            return (today + timedelta(days=days_ahead)).strftime('%Y-%m-%d')

        # Try YYYY-MM-DD format
        if re.match(r'^\d{4}-\d{2}-\d{2}$', date_input):
            date_obj = datetime.strptime(date_input, '%Y-%m-%d').date()
            if date_obj < today:
                raise ValueError("Date cannot be in the past")
            return date_input

        # Try other formats
        formats = ['%d/%m/%Y', '%m/%d/%Y', '%d-%m-%Y']
        for fmt in formats:
            try:
                date_obj = datetime.strptime(date_input, fmt).date()
                if date_obj < today:
                    raise ValueError("Date cannot be in the past")
                return date_obj.strftime('%Y-%m-%d')
            except:
                continue

        raise ValueError(
            "Use format like 'YYYY-MM-DD', 'tomorrow', or 'next Monday'")

    def _parse_time(self, time_input):
        """Parse time in various formats"""
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

        raise ValueError("Use format like '10:30', '2:00 PM', or '14:00'")

    def _complete_callback(self):
        """Complete callback booking"""
        contact = ContactInfo(
            name=self.data['name'],
            phone=self.data['phone'],
            email=self.data['email']
        )

        result = f"""✅ **Callback Request Submitted!**

📋 **Details:**
• **Name:** {contact.name}
• **Phone:** {contact.phone}
• **Email:** {contact.email}

🔄 **Processing booking...**"""

        self.reset_form()
        return result

    def _complete_appointment(self):
        """Complete appointment booking"""
        contact = ContactInfo(
            name=self.data['name'],
            phone=self.data['phone'],
            email=self.data['email']
        )

        appointment = AppointmentDetails(
            contact=contact,
            date=self.data['date'],
            time=self.data['time'],
            purpose=self.data['purpose']
        )

        result = f"""✅ **Appointment Booking Submitted!**

📋 **Contact Details:**
• **Name:** {appointment.contact.name}
• **Phone:** {appointment.contact.phone}
• **Email:** {appointment.contact.email}

📅 **Appointment Details:**
• **Date:** {appointment.date}
• **Time:** {appointment.time}
• **Purpose:** {appointment.purpose}

🔄 **Processing booking...**"""

        self.reset_form()
        return result

    def is_active(self):
        """Check if form is currently active"""
        return self.form_type is not None

    def cancel(self):
        """Cancel current form"""
        self.reset_form()
        return "❌ Booking cancelled. How else can I help you?"

    def get_form_data(self):
        """Get current form data"""
        return self.data.copy()

    def get_booking_type(self):
        """Get current booking type"""
        return self.form_type
