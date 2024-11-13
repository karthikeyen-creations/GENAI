import streamlit as st

# Conditionally disabled text area (based on main_input's content, which appears after it)
text_area_disabled = not bool(st.session_state.get("main_input", ""))  # Default to empty if main_input isn't set yet
text_input = st.text_area("Enter your text:", value=st.session_state.get("main_input", ""), disabled=text_area_disabled)

# Primary text input to check for content (appears after the text area)
main_input = st.text_area("Type here to disable the text area:", key="main_input")

# Display the content entered in the text area
st.write("You entered:", text_input)
