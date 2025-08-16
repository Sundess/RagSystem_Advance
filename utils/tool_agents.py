from typing import Dict, Any, List
import json
import streamlit as st
import time
from pydantic import BaseModel


class BookingTool(BaseModel):
    """Tool for handling booking operations"""
    name: str = "booking_tool"
    description: str = "Handles appointment and callback bookings"

    def execute(self, action: str, data: Dict[str, Any]) -> str:
        """Execute booking action"""
        if action == "callback":
            return self._process_callback(data)
        elif action == "appointment":
            return self._process_appointment(data)
        else:
            return f"❌ Unknown booking action: {action}"

    def _process_callback(self, data: Dict[str, Any]) -> str:
        """Process callback booking"""
        st.info("📞 **Processing Callback Request...**")

        # Show booking progress
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i in range(101):
            progress_bar.progress(i)
            if i < 30:
                status_text.text("🔍 Validating contact information...")
            elif i < 60:
                status_text.text("📝 Creating callback ticket...")
            elif i < 90:
                status_text.text("📤 Sending confirmation...")
            else:
                status_text.text("✅ Finalizing booking...")

            time.sleep(0.1)  # 10 seconds total

        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()

        return f"""🎉 **Callback Booked Successfully!**

📞 **Confirmation Details:**
- **Reference ID:** CB-{int(time.time())}
- **Name:** {data.get('name', 'N/A')}
- **Phone:** {data.get('phone', 'N/A')}
- **Email:** {data.get('email', 'N/A')}

📧 A confirmation email has been sent to your email address.
📱 You'll receive a call within 24-48 hours.

Thank you for choosing our services! 🙏"""

    def _process_appointment(self, data: Dict[str, Any]) -> str:
        """Process appointment booking"""
        st.info("📅 **Processing Appointment Booking...**")

        # Show booking progress
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i in range(101):
            progress_bar.progress(i)
            if i < 20:
                status_text.text("🔍 Validating appointment details...")
            elif i < 40:
                status_text.text("📅 Checking calendar availability...")
            elif i < 60:
                status_text.text("📝 Creating appointment slot...")
            elif i < 80:
                status_text.text("📤 Sending calendar invite...")
            else:
                status_text.text("✅ Finalizing appointment...")

            time.sleep(0.1)  # 10 seconds total

        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()

        return f"""🎉 **Appointment Booked Successfully!**

📅 **Confirmation Details:**
- **Reference ID:** APT-{int(time.time())}
- **Name:** {data.get('name', 'N/A')}
- **Phone:** {data.get('phone', 'N/A')}
- **Email:** {data.get('email', 'N/A')}
- **Date:** {data.get('date', 'N/A')}
- **Time:** {data.get('time', 'N/A')}
- **Purpose:** {data.get('purpose', 'N/A')}

📧 Calendar invite sent to your email address.
⏰ You'll receive a reminder 24 hours before the appointment.

Looking forward to meeting with you! 🤝"""


class ToolAgent:
    """Agent that manages and executes various tools"""

    def __init__(self):
        self.tools = {
            "booking": BookingTool()
        }

    def detect_tool_call(self, message: str) -> tuple:
        """Detect if message requires a tool call"""
        message_lower = message.lower()

        # Booking tool triggers
        booking_triggers = [
            'book appointment', 'schedule appointment', 'book meeting',
            'call me back', 'callback', 'schedule call', 'arrange call',
            'book a call', 'set up meeting', 'make appointment'
        ]

        for trigger in booking_triggers:
            if trigger in message_lower:
                return ("booking", self._determine_booking_action(message_lower))

        return (None, None)

    def _determine_booking_action(self, message: str) -> str:
        """Determine specific booking action"""
        callback_keywords = ['call me', 'callback', 'call back', 'phone me']

        if any(keyword in message for keyword in callback_keywords):
            return "callback"
        else:
            return "appointment"

    def execute_tool(self, tool_name: str, action: str, data: Dict[str, Any]) -> str:
        """Execute a specific tool with given data"""
        if tool_name not in self.tools:
            return f"❌ Tool '{tool_name}' not found"

        try:
            return self.tools[tool_name].execute(action, data)
        except Exception as e:
            return f"❌ Error executing {tool_name}: {str(e)}"

    def list_available_tools(self) -> List[str]:
        """List all available tools"""
        return list(self.tools.keys())

    def get_tool_info(self, tool_name: str) -> str:
        """Get information about a specific tool"""
        if tool_name in self.tools:
            return self.tools[tool_name].description
        return f"Tool '{tool_name}' not found"


class BookingAgent:
    """Specialized agent for handling booking operations"""

    def __init__(self, conversation_form, tool_agent):
        self.conversation_form = conversation_form
        self.tool_agent = tool_agent

    def process_message(self, message: str) -> tuple:
        """Process message and determine if booking is needed"""

        # Check if form is active
        if self.conversation_form.is_form_active():
            form_response = self.conversation_form.process_form_input(message)

            # Check if form is completed (contains booking confirmation)
            if form_response and "Processing booking..." in form_response:
                # Extract data and trigger tool
                form_data = self._extract_form_data()
                action = self._get_booking_action()

                # Execute tool
                tool_response = self.tool_agent.execute_tool(
                    "booking", action, form_data)

                return (form_response + "\n\n" + tool_response, True)

            return (form_response, False)

        # Detect new booking intent
        if self.conversation_form.detect_booking_intent(message):
            tool_name, action = self.tool_agent.detect_tool_call(message)

            if action == "callback":
                response = self.conversation_form.start_contact_form()
            else:
                response = self.conversation_form.start_appointment_form()

            return (response, False)

        return (None, False)

    def _extract_form_data(self) -> Dict[str, Any]:
        """Extract data from completed form"""
        return self.conversation_form.form_data

    def _get_booking_action(self) -> str:
        """Get the current booking action"""
        if self.conversation_form.current_form == "contact":
            return "callback"
        else:
            return "appointment"

    def cancel_current_booking(self) -> str:
        """Cancel current booking process"""
        return self.conversation_form.cancel_form()

    def is_booking_active(self) -> bool:
        """Check if booking process is active"""
        return self.conversation_form.is_form_active()
