import streamlit as st
from PIL import Image, ImageGrab
import pytesseract

# Set up the title
st.title("Dynamic OCR with Editable Text Inputs and Stitching")

# Description
st.markdown("""
Dynamically add input slots to upload or paste images. Each slot displays the recognized text individually, which can be edited. The final stitched text removes duplicates only at the boundaries between consecutive images.
""")

# Initialize session state for dynamic image slots
if "image_slots" not in st.session_state:
    st.session_state.image_slots = []  # Stores the images
    st.session_state.texts = []  # Stores extracted text for each slot
    st.session_state.num_slots = 1  # Start with one input slot

# Add a new slot dynamically
def add_new_slot():
    st.session_state.num_slots += 1
    st.session_state.image_slots.append(None)  # Add a placeholder for the new image slot
    st.session_state.texts.append("")  # Add a placeholder for the new extracted text

# Function to remove overlaps only at boundaries
def remove_overlap_and_concatenate(*texts):
    """Concatenate texts while removing overlaps at boundaries."""
    if not texts:
        return ""
    result = texts[0]
    for i in range(1, len(texts)):
        max_overlap = 0
        min_length = min(len(result), len(texts[i]))
        for j in range(1, min_length + 1):
            if result[-j:] == texts[i][:j]:
                max_overlap = j
        result += texts[i][max_overlap:]
    return result

# Callback to update text in session state when edited
def update_text(idx):
    """Update session state for a specific text slot when edited."""
    st.session_state.texts[idx] = st.session_state[f"text_area_{idx}"]

# Render input slots
for idx in range(st.session_state.num_slots):
    st.markdown(f"### Input Slot {idx + 1}")
    col1, col2, col3 = st.columns([3, 3, 1])

    # Upload an image
    with col1:
        uploaded_file = st.file_uploader(f"Upload Image {idx + 1}", type=["png", "jpg", "jpeg"], key=f"upload_{idx}")
        if uploaded_file:
            try:
                image = Image.open(uploaded_file).convert("RGB")
                if len(st.session_state.image_slots) <= idx:
                    st.session_state.image_slots.append(image)
                else:
                    st.session_state.image_slots[idx] = image
                st.success(f"Image {idx + 1} uploaded successfully!")
            except Exception as e:
                st.error(f"Error processing uploaded image: {e}")

    # Paste an image from clipboard
    with col2:
        if st.button(f"Paste Image {idx + 1}", key=f"paste_{idx}"):
            try:
                clipboard_image = ImageGrab.grabclipboard()
                if isinstance(clipboard_image, Image.Image):
                    clipboard_image = clipboard_image.convert("RGB")
                    if len(st.session_state.image_slots) <= idx:
                        st.session_state.image_slots.append(clipboard_image)
                    else:
                        st.session_state.image_slots[idx] = clipboard_image
                    st.success(f"Image {idx + 1} pasted successfully!")
                else:
                    st.error("No valid image found in the clipboard. Please copy an image and try again.")
            except Exception as e:
                st.error(f"Failed to paste image: {e}")

    # Display the selected image
    if idx < len(st.session_state.image_slots) and st.session_state.image_slots[idx]:
        st.image(st.session_state.image_slots[idx], caption=f"Image {idx + 1}", use_column_width=True)

    # Regenerate extracted text
    with col3:
        if st.button(f"Regenerate", key=f"regen_{idx}"):
            if idx < len(st.session_state.image_slots) and st.session_state.image_slots[idx]:
                try:
                    regenerated_text = pytesseract.image_to_string(st.session_state.image_slots[idx]).strip()
                    st.session_state.texts[idx] = regenerated_text
                    st.success(f"Text for Image {idx + 1} regenerated!")
                except Exception as e:
                    st.error(f"Error regenerating text for Image {idx + 1}: {e}")
            else:
                st.error("No image provided for this slot!")

    # Ensure `st.session_state.texts` matches the number of slots
    while len(st.session_state.texts) < st.session_state.num_slots:
        st.session_state.texts.append("")  # Add empty placeholders for new slots

    # Extract and display recognized text for this slot
    st.markdown(f"#### Recognized Text for Input Slot {idx + 1}")
    if idx < len(st.session_state.image_slots) and st.session_state.image_slots[idx]:
        if idx >= len(st.session_state.texts) or not st.session_state.texts[idx]:
            try:
                st.session_state.texts[idx] = pytesseract.image_to_string(st.session_state.image_slots[idx]).strip()
            except Exception as e:
                st.error(f"Error extracting text from Image {idx + 1}: {e}")
        # Add `text_area` with `on_change` callback
        st.text_area(
            f"Recognized Text (Slot {idx + 1})",
            st.session_state.texts[idx],
            height=200,
            key=f"text_area_{idx}",
            on_change=update_text,
            args=(idx,)  # Pass index as argument
        )

# Add the "Add New Image Input" button at the bottom
if st.button("Add New Image Input", key="add_input"):
    add_new_slot()

# Process all provided images
if any(st.session_state.image_slots):
    st.markdown("### Extracted and Stitched Text")
    # Stitch texts using remove_overlap_and_concatenate
    stitched_text = remove_overlap_and_concatenate(*st.session_state.texts)

    # Display stitched text
    st.text_area("Stitched Text (Boundary Overlaps Removed)", stitched_text, height=500)

    # Copy stitched text to clipboard
    if st.button("Copy Stitched Text to Clipboard"):
        import pyperclip

        try:
            pyperclip.copy(stitched_text)
            st.success("Stitched text copied to clipboard!")
        except Exception as e:
            st.error(f"Failed to copy text: {e}")
else:
    st.info("Provide at least one image to begin.")
