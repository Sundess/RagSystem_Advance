from langchain.tools import BaseTool
from langchain.callbacks.manager import CallbackManagerForToolRun
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, Type
import streamlit as st
import time
import json
import re
from datetime import datetime, timedelta


class BookingInput(BaseModel):
    """Input schema for booking tool"""
    booking_type: str = Field(
        description="Type of booking: 'appointment' or 'callback'")
    contact_data: Dict[str, Any] = Field(
        description="Contact information dictionary")


class InputParser:
    """Enhanced input parser using LLM for natural language processing"""

    def __init__(self, gemini_chat):
        self.gemini_chat = gemini_chat
        self.current_date = datetime.now()

    def parse_date(self, user_input: str) -> tuple[str, str]:
        """Parse date using LLM and return (formatted_date, error_message)"""
        current_date_str = self.current_date.strftime("%Y-%m-%d")
        current_day = self.current_date.strftime("%A")

        prompt = f"""
You are a date parser. Current date is {current_date_str} ({current_day}).

Parse the user's date input and return ONLY a valid date in YYYY-MM-DD format.
If the input is invalid or unclear, return "INVALID".

Examples:
- "today" -> {current_date_str}
- "tomorrow" -> {(self.current_date + timedelta(days=1)).strftime("%Y-%m-%d")}
- "next Monday" -> (calculate next Monday from current date)
- "December 25" -> 2024-12-25 (assume current year if not specified)
- "12/25/2024" -> 2024-12-25
- "25th" -> INVALID (too vague)

User input: "{user_input}"

Response (YYYY-MM-DD format only):
"""

        try:
            response = self.gemini_chat.generate_simple_answer(prompt).strip()

            # Validate the response format
            if re.match(r'^\d{4}-\d{2}-\d{2}$', response):
                # Check if date is not in the past
                parsed_date = datetime.strptime(response, '%Y-%m-%d').date()
                if parsed_date < self.current_date.date():
                    return "", "Date cannot be in the past. Please choose a future date."
                return response, ""
            else:
                return "", "Please provide a clearer date (e.g., 'tomorrow', 'next Monday', '2024-12-25')"

        except Exception as e:
            return "", "Error parsing date. Please try again with a clearer format."

    def parse_time(self, user_input: str) -> tuple[str, str]:
        """Parse time using LLM and return (formatted_time, error_message)"""
        prompt = f"""
You are a time parser. Parse the user's time input and return ONLY a valid time in HH:MM AM/PM format (12-hour format).
If the input is invalid or unclear, return "INVALID".

Examples:
- "3:40 afternoon" -> 3:40 PM
- "around 3:40 afternoon" -> 3:40 PM
- "10:30" -> 10:30 AM
- "2 PM" -> 2:00 PM
- "14:00" -> 2:00 PM
- "morning 9" -> 9:00 AM
- "evening 7" -> 7:00 PM
- "noon" -> 12:00 PM
- "midnight" -> 12:00 AM

User input: "{user_input}"

Response (HH:MM AM/PM format only):
"""

        try:
            response = self.gemini_chat.generate_simple_answer(prompt).strip()

            # Validate the response format (HH:MM AM/PM)
            if re.match(r'^\d{1,2}:\d{2}\s?(AM|PM)$', response, re.IGNORECASE):
                return response, ""
            else:
                return "", "Please provide a clearer time (e.g., '3:40 PM', '10:30 AM', '2 PM')"

        except Exception as e:
            return "", "Error parsing time. Please try again with a clearer format."

    def validate_email(self, email: str) -> tuple[str, str]:
        """Validate and format email"""
        email = email.strip().lower()

        # Basic email validation
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if re.match(email_pattern, email):
            return email, ""
        else:
            return "", "Please provide a valid email address (e.g., john@example.com)"

    def validate_phone(self, phone: str) -> tuple[str, str]:
        """Validate and format phone number"""
        # Remove all non-digit characters
        digits = re.sub(r'\D', '', phone)

        if len(digits) == 10:
            # Format as (XXX) XXX-XXXX
            formatted = f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
            return formatted, ""
        elif len(digits) == 11 and digits.startswith('1'):
            # Handle US number with country code
            digits = digits[1:]
            formatted = f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
            return formatted, ""
        else:
            return "", "Please provide a valid 10-digit phone number"

    def validate_name(self, name: str) -> tuple[str, str]:
        """Validate and format name"""
        name = name.strip()

        if len(name) < 2:
            return "", "Please provide your full name (at least 2 characters)"

        if not re.match(r'^[a-zA-Z\s\'-]+$', name):
            return "", "Please provide a valid name (letters, spaces, hyphens, and apostrophes only)"

        # Format name (title case)
        formatted_name = ' '.join(word.capitalize() for word in name.split())
        return formatted_name, ""


class BookingTool(BaseTool):
    """Enhanced LangChain tool for handling bookings"""

    name: str = "booking_processor"
    description: str = """
    Process appointment and callback bookings. 
    Input should be a dictionary with 'booking_type' (appointment/callback) and 'contact_data'.
    """
    args_schema: Type[BaseModel] = BookingInput

    def _run(
        self,
        booking_type: str,
        contact_data: Dict[str, Any],
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Execute booking process"""

        if booking_type == "callback":
            return self._process_callback(contact_data)
        elif booking_type == "appointment":
            return self._process_appointment(contact_data)
        else:
            return f"❌ Unknown booking type: {booking_type}"

    def _process_callback(self, data: Dict[str, Any]) -> str:
        """Process callback booking"""
        # Show progress
        self._show_progress("callback")

        ref_id = f"CB-{int(time.time())}"

        return json.dumps({
            "status": "success",
            "type": "callback",
            "reference_id": ref_id,
            "message": f"""🎉 **Callback Booked Successfully!**

📞 **Contact Details:**
• **Reference:** {ref_id}
• **Name:** {data.get('name', 'N/A')}
• **Phone:** {data.get('phone', 'N/A')}
• **Email:** {data.get('email', 'N/A')}

📧 **Confirmation email sent to {data.get('email', 'your email')}**
📱 **You'll receive a call within 24-48 hours**

Thank you for choosing our services!"""
        })

    def _process_appointment(self, data: Dict[str, Any]) -> str:
        """Process appointment booking"""
        # Show progress
        self._show_progress("appointment")

        ref_id = f"APT-{int(time.time())}"

        return json.dumps({
            "status": "success",
            "type": "appointment",
            "reference_id": ref_id,
            "message": f"""🎉 **Appointment Booked Successfully!**

📅 **Appointment Details:**
• **Reference:** {ref_id}
• **Name:** {data.get('name', 'N/A')}
• **Phone:** {data.get('phone', 'N/A')}
• **Email:** {data.get('email', 'N/A')}
• **Date:** {data.get('date', 'N/A')}
• **Time:** {data.get('time', 'N/A')}
• **Purpose:** {data.get('purpose', 'N/A')}

📧 **Calendar invite sent to {data.get('email', 'your email')}**
⏰ **Reminder set for 24 hours before your appointment**
🗓️ **Scheduled for {data.get('date', 'N/A')} at {data.get('time', 'N/A')}**

Thank you for booking with us!"""
        })

    def _show_progress(self, booking_type: str):
        """Show booking progress"""
        steps = [
            "🔍 Validating information...",
            "📝 Creating booking...",
            "📤 Sending confirmation...",
            "✅ Finalizing..."
        ]

        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, step in enumerate(steps):
            progress = int((i + 1) * 100 / len(steps))
            progress_bar.progress(progress)
            status_text.text(step)
            time.sleep(1.2)

        progress_bar.empty()
        status_text.empty()


class EnhancedBookingAgent:
    """Enhanced booking agent with LLM-powered parsing and validation"""

    def __init__(self, gemini_chat):
        self.gemini_chat = gemini_chat
        self.parser = InputParser(gemini_chat)
        self.booking_tool = BookingTool()
        self.current_booking = None
        self.form_data = {}
        self.form_step = 0
        self.booking_steps = {
            'callback': ['name', 'phone', 'email'],
            'appointment': ['name', 'phone', 'email', 'date', 'time', 'purpose']
        }

    def process_message(self, message: str) -> tuple:
        """Process user message for booking with enhanced parsing"""

        print(f"🤖 EnhancedBookingAgent processing: '{message}'")
        print(f"🤖 Current booking type: {self.current_booking}")
        print(f"🤖 Form step: {self.form_step}")
        print(f"🤖 Form data: {self.form_data}")

        # Check if currently in booking flow
        if self.current_booking:
            print("📝 Processing form step...")
            return self._handle_form_step(message)

        # Detect new booking intent
        if self._detect_booking_intent(message):
            booking_type = self._determine_booking_type(message)
            self.current_booking = booking_type
            self.form_step = 0
            self.form_data = {}

            print(f"📞 Starting {booking_type} booking...")

            if booking_type == 'callback':
                return ("I'll arrange a callback for you! 📞\n\n**What's your full name?**", False)
            else:
                return ("Let's book your appointment! 📅\n\n**What's your full name?**", False)

        print("❌ No booking intent detected")
        return (None, False)

    def _detect_booking_intent(self, message: str) -> bool:
        """Enhanced booking intent detection"""
        booking_keywords = [
            'book', 'schedule', 'appointment', 'meeting', 'callback',
            'call me', 'arrange', 'set up', 'reserve', 'appoint',
            'call back', 'contact me', 'book me', 'book an', 'schedule a'
        ]
        message_lower = message.lower().strip()

        # Check for exact matches and partial matches
        detected = any(
            keyword in message_lower for keyword in booking_keywords)

        # Additional checks for common booking phrases
        booking_phrases = [
            'i want to book', 'i need to schedule', 'can you book',
            'i would like to', 'please book', 'please schedule'
        ]

        if not detected:
            detected = any(
                phrase in message_lower for phrase in booking_phrases)

        print(f"🔍 Booking intent detection for '{message}': {detected}")
        return detected

    def _determine_booking_type(self, message: str) -> str:
        """Determine if callback or appointment"""
        callback_keywords = ['call me', 'callback',
                             'call back', 'phone me', 'ring me']
        message_lower = message.lower()

        if any(keyword in message_lower for keyword in callback_keywords):
            return 'callback'
        return 'appointment'

    def _handle_form_step(self, user_input: str) -> tuple:
        """Handle current form step with enhanced validation"""
        if self.form_step >= len(self.booking_steps[self.current_booking]):
            print("⚠️ Form step out of range, resetting...")
            self._reset_form()
            return ("Something went wrong. Let's start over. How can I help you?", False)

        current_step = self.booking_steps[self.current_booking][self.form_step]
        print(f"📝 Handling step '{current_step}' with input: '{user_input}'")

        # Validate and parse the input
        parsed_value, error_message = self._validate_and_parse_input(
            current_step, user_input.strip())

        if error_message:
            # Validation failed, ask again with error message
            return (f"❌ {error_message}\n\n{self._get_step_question(current_step)}", False)

        # Store the valid input
        self.form_data[current_step] = parsed_value
        self.form_step += 1

        print(f"📝 Updated form data: {self.form_data}")
        print(
            f"📝 Next step: {self.form_step}/{len(self.booking_steps[self.current_booking])}")

        # Check if form is complete
        if self.form_step >= len(self.booking_steps[self.current_booking]):
            print("✅ Form complete, processing booking...")
            return self._complete_booking()

        # Ask next question
        return self._get_next_question()

    def _validate_and_parse_input(self, field: str, value: str) -> tuple[str, str]:
        """Validate and parse user input using LLM for complex fields"""
        if not value or len(value.strip()) == 0:
            return "", "Please provide a valid response."

        if field == 'name':
            return self.parser.validate_name(value)

        elif field == 'phone':
            return self.parser.validate_phone(value)

        elif field == 'email':
            return self.parser.validate_email(value)

        elif field == 'date':
            return self.parser.parse_date(value)

        elif field == 'time':
            return self.parser.parse_time(value)

        elif field == 'purpose':
            if len(value) < 5:
                return "", "Please provide more details about the appointment purpose (at least 5 characters)."
            return value.strip(), ""

        return value.strip(), ""

    def _get_step_question(self, step: str) -> str:
        """Get question text for a specific step"""
        questions = {
            'name': "**What's your full name?**",
            'phone': "**What's your phone number?** (e.g., (555) 123-4567)",
            'email': "**What's your email address?** (e.g., john@example.com)",
            'date': "**When would you like the appointment?**\n(e.g., 'tomorrow', 'next Monday', '2024-12-25', 'today')",
            'time': "**What time would you prefer?**\n(e.g., '3:40 PM', 'around 2 afternoon', '10:30 AM')",
            'purpose': "**What's the purpose of the appointment?**\n(Please provide some details)"
        }
        return questions.get(step, "Please continue...")

    def _get_next_question(self) -> tuple:
        """Get next question in the form"""
        if self.form_step >= len(self.booking_steps[self.current_booking]):
            return ("Form completed!", True)

        next_step = self.booking_steps[self.current_booking][self.form_step]
        question = self._get_step_question(next_step)

        # Add confirmation for previous step
        prev_step = self.booking_steps[self.current_booking][self.form_step - 1]
        prev_value = self.form_data.get(prev_step, '')

        confirmation = f"✅ **{prev_step.title()}:** {prev_value}\n\n{question}"

        return (confirmation, False)

    def _complete_booking(self) -> tuple:
        """Complete the booking using LangChain tool"""
        try:
            print(
                f"🎯 Completing {self.current_booking} booking with data: {self.form_data}")

            # Show final summary before processing
            summary = self._generate_booking_summary()

            # Use the LangChain tool to process booking
            result = self.booking_tool._run(
                booking_type=self.current_booking,
                contact_data=self.form_data
            )

            # Parse result
            booking_result = json.loads(result)
            response = booking_result['message']

            # Reset form
            self._reset_form()

            return (response, True)

        except Exception as e:
            print(f"❌ Error completing booking: {e}")
            self._reset_form()
            return (f"❌ Error completing booking: {str(e)}", False)

    def _generate_booking_summary(self) -> str:
        """Generate a summary of the booking before processing"""
        if self.current_booking == 'callback':
            return f"""
📋 **Callback Summary:**
• Name: {self.form_data.get('name', '')}
• Phone: {self.form_data.get('phone', '')}
• Email: {self.form_data.get('email', '')}
"""
        else:
            return f"""
📋 **Appointment Summary:**
• Name: {self.form_data.get('name', '')}
• Phone: {self.form_data.get('phone', '')}
• Email: {self.form_data.get('email', '')}
• Date: {self.form_data.get('date', '')}
• Time: {self.form_data.get('time', '')}
• Purpose: {self.form_data.get('purpose', '')}
"""

    def _reset_form(self):
        """Reset booking form"""
        print("🔄 Resetting booking form...")
        self.current_booking = None
        self.form_data = {}
        self.form_step = 0

    def is_booking_active(self) -> bool:
        """Check if booking is active"""
        active = self.current_booking is not None
        print(
            f"🔍 Booking active check: {active} (type: {self.current_booking})")
        return active

    def cancel_booking(self) -> str:
        """Cancel current booking"""
        if self.current_booking:
            booking_type = self.current_booking
            self._reset_form()
            return f"❌ {booking_type.title()} booking cancelled. How else can I help you?"
        return "No active booking to cancel."

    def get_current_step(self) -> str:
        """Get current step for debugging"""
        if not self.current_booking or self.form_step >= len(self.booking_steps[self.current_booking]):
            return "No active step"
        return self.booking_steps[self.current_booking][self.form_step]

    def get_booking_progress(self) -> str:
        """Get booking progress for UI display"""
        if not self.current_booking:
            return "No active booking"

        total_steps = len(self.booking_steps[self.current_booking])
        return f"Step {self.form_step + 1}/{total_steps}: {self.get_current_step()}"


# Update the SimplifiedBookingAgent alias for backward compatibility
SimplifiedBookingAgent = EnhancedBookingAgent
