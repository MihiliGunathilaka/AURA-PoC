import streamlit as st

# Email templates

def _default_email_templates(predicted_state: str, user_id: str):
    if predicted_state == "At-Risk":
        subject = "We noticed reduced activity — can we help you get more value?"
        body = f"""Hi {user_id},

We noticed a significant drop in your recent usage on the platform.

We’d love to help you re-engage:
- Quick support call / chat if needed
- Tips based on features you use less
- Optional limited-time offer

Best regards,
AURA Retention Team
"""
    else:
        subject = "Quick tips to help you get more from your cloud platform"
        body = f"""Hi {user_id},

We noticed a slight decrease in recent engagement.

Here are some quick suggestions:
- Feature reminders
- Short tutorials
- Guided walkthroughs

Best regards,
AURA Support Team
"""
    return subject, body


# Main popup renderer

def render_email_draft_popup(*, predicted_state: str, user_id: str, to_email_default: str):

    # Session state flags
    st.session_state.setdefault("show_email_popup", False)
    st.session_state.setdefault("show_email_confirmation", False)
    st.session_state.setdefault("email_confirmation_data", None)

    has_dialog = hasattr(st, "dialog")

    # DRAFT FORM
    
    def _render_draft():
        if "email_subject" not in st.session_state:
            subject, body = _default_email_templates(predicted_state, user_id)
            st.session_state.email_subject = subject
            st.session_state.email_body = body
            st.session_state.email_to = to_email_default

        st.write(f"**Risk state:** `{predicted_state}`")
        st.text_input("To", key="email_to")
        st.text_input("Subject", key="email_subject")
        st.text_area("Message", key="email_body", height=220)

        col1, col2 = st.columns(2)

        # SEND
        with col1:
            if st.button("Send"):
                st.session_state.email_confirmation_data = {
                    "to": st.session_state.email_to,
                    "subject": st.session_state.email_subject,
                    "user_id": user_id,
                    "risk_state": predicted_state,
                }

                # Close draft, open confirmation
                st.session_state.show_email_popup = False
                st.session_state.show_email_confirmation = True

                # Cleanup
                for k in ["email_subject", "email_body", "email_to"]:
                    st.session_state.pop(k, None)

                st.rerun()

        # CLOSE
        with col2:
            if st.button("Close"):
                st.session_state.show_email_popup = False
                for k in ["email_subject", "email_body", "email_to"]:
                    st.session_state.pop(k, None)
                st.rerun()

    # CONFIRMATION
    
    def _render_confirmation():
        data = st.session_state.email_confirmation_data

        st.success("Email sent successfully (PoC Simulation)")
        st.write(f"**To:** {data['to']}")
        st.write(f"**Subject:** {data['subject']}")
        st.write(f"**User:** {data['user_id']}")
        st.write(f"**Risk state:** {data['risk_state']}")

        if st.button("OK"):
            st.session_state.show_email_confirmation = False
            st.session_state.email_confirmation_data = None
            st.rerun()

    # DISPLAY LOGIC (ONLY ONE DIALOG PER RUN)
    
    if st.session_state.show_email_popup:
        if has_dialog:
            @st.dialog("Engagement Email Draft")
            def _dlg():
                _render_draft()
            _dlg()
        else:
            _render_draft()

  
