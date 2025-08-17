from langchain.tools import BaseTool
from langchain.callbacks.manager import CallbackManagerForToolRun
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, Type
import streamlit as st
import time
import json


class BookingInput(BaseModel):
    """Input schema for booking tool"""
    booking_type: str = Field(
        description="Type of booking: 'appointment' or 'callback'")
    contact_data: Dict[str, Any] = Field(
        description="Contact information dictionary")


class BookingTool(BaseTool):
    """LangChain tool for handling bookings"""

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

📞 **Details:**
• **Reference:** {ref_id}
• **Name:** {data.get('name', 'N/A')}
• **Phone:** {data.get('phone', 'N/A')}
• **Email:** {data.get('email', 'N/A')}

📧 Confirmation email sent!
📱 You'll receive a call within 24-48 hours."""
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

📅 **Details:**
• **Reference:** {ref_id}
• **Name:** {data.get('name', 'N/A')}
• **Phone:** {data.get('phone', 'N/A')}
• **Email:** {data.get('email', 'N/A')}
• **Date:** {data.get('date', 'N/A')}
• **Time:** {data.get('time', 'N/A')}
• **Purpose:** {data.get('purpose', 'N/A')}

📧 Calendar invite sent!
⏰ Reminder set for 24 hours before."""
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
            time.sleep(1.5)

        progress_bar.empty()
        status_text.empty()


class SimplifiedBookingAgent:
    """Simplified booking agent using LangChain tools"""

    def __init__(self, gemini_chat):
        self.gemini_chat = gemini_chat
        self.booking_tool = BookingTool()
        self.current_booking = None
        self.form_data = {}
        self.form_step = 0
        self.booking_steps = {
            'callback': ['name', 'phone', 'email'],
            'appointment': ['name', 'phone', 'email', 'date', 'time', 'purpose']
        }

    def process_message(self, message: str) -> tuple:
        """Process user message for booking"""

        print(f"🤖 BookingAgent processing: '{message}'")
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
                return ("I'll arrange a callback! 📞\n\n**What's your name?**", False)
            else:
                return ("Let's book your appointment! 📅\n\n**What's your name?**", False)

        print("❌ No booking intent detected")
        return (None, False)

    def _detect_booking_intent(self, message: str) -> bool:
        """Simple keyword-based intent detection"""
        booking_keywords = [
            'book', 'schedule', 'appointment', 'meeting', 'callback',
            'call me', 'arrange', 'set up', 'reserve', 'appoint'
        ]
        message_lower = message.lower()
        detected = any(
            keyword in message_lower for keyword in booking_keywords)
        print(f"🔍 Booking intent detection for '{message}': {detected}")
        return detected

    def _determine_booking_type(self, message: str) -> str:
        """Determine if callback or appointment"""
        callback_keywords = ['call me', 'callback', 'call back', 'phone']
        message_lower = message.lower()

        if any(keyword in message_lower for keyword in callback_keywords):
            return 'callback'
        return 'appointment'

    def _handle_form_step(self, user_input: str) -> tuple:
        """Handle current form step"""
        current_step = self.booking_steps[self.current_booking][self.form_step]

        print(f"📝 Handling step '{current_step}' with input: '{user_input}'")

        # Store the input
        self.form_data[current_step] = user_input.strip()
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

    def _get_next_question(self) -> tuple:
        """Get next question in the form"""
        next_step = self.booking_steps[self.current_booking][self.form_step]

        questions = {
            'name': "**What's your name?**",
            'phone': "**What's your phone number?**",
            'email': "**What's your email address?**",
            'date': "**When would you like the appointment?** (e.g., '2024-12-25')",
            'time': "**What time?** (e.g., '10:30 AM')",
            'purpose': "**What's the purpose of the appointment?**"
        }

        return (questions.get(next_step, "Please continue..."), False)

    def _complete_booking(self) -> tuple:
        """Complete the booking using LangChain tool"""
        try:
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
            self._reset_form()
            return (f"❌ Error completing booking: {str(e)}", False)

    def _reset_form(self):
        """Reset booking form"""
        self.current_booking = None
        self.form_data = {}
        self.form_step = 0

    def is_booking_active(self) -> bool:
        """Check if booking is active"""
        return self.current_booking is not None

    def cancel_booking(self) -> str:
        """Cancel current booking"""
        self._reset_form()
        return "❌ Booking cancelled. How else can I help you?"
