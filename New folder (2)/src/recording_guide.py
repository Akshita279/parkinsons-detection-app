#!/usr/bin/env python3
"""
Clinical Voice Recording Protocol for Parkinson's Detection
Based on research from Parkinson's Voice Initiative and clinical studies
"""

import streamlit as st
import time
import numpy as np
from pathlib import Path

def create_recording_guide():
    """Create comprehensive recording guide"""
    
    guide = {
        "clinical_protocol": {
            "primary_task": "Sustained Phonation",
            "vowel_sounds": ["'Ahhhh' (/a/)", "'Ohhh' (/o/)", "'Ehhh' (/e/)"],
            "duration": "3-15 seconds per recording",
            "repetitions": "3 times per vowel",
            "total_recordings": "9 recordings minimum"
        },
        
        "recording_specifications": {
            "sample_rate": "≥16 kHz (preferably 44.1 kHz)",
            "bit_depth": "16-bit minimum (32-bit preferred)",
            "format": "WAV (uncompressed)",
            "microphone_distance": "15-20 cm from mouth",
            "environment": "Quiet room (<40 dB background noise)"
        },
        
        "clinical_instructions": {
            "preparation": [
                "Sit upright in comfortable position",
                "Remove any jewelry that might create noise",
                "Clear throat gently before recording",
                "Take normal breath before each recording"
            ],
            
            "execution": [
                "Take a deep breath",
                "Say the vowel sound as steadily as possible",
                "Maintain consistent volume (comfortable speaking level)",
                "Keep pitch as steady as possible",
                "Continue until you run out of breath naturally",
                "Don't force or strain your voice"
            ],
            
            "between_recordings": [
                "Rest 10-15 seconds between recordings",
                "Breathe normally",
                "Swallow if needed to clear throat"
            ]
        },
        
        "additional_tasks": {
            "diadochokinetic": [
                "Repeat 'pa-pa-pa' as fast as possible for 10 seconds",
                "Repeat 'ta-ta-ta' as fast as possible for 10 seconds", 
                "Repeat 'ka-ka-ka' as fast as possible for 10 seconds",
                "Repeat 'pa-ta-ka' sequence 10 times"
            ],
            
            "reading_tasks": [
                "Read 'The Rainbow Passage' (standard speech sample)",
                "Count from 1 to 20 at normal pace",
                "Recite days of the week"
            ],
            
            "spontaneous_speech": [
                "Describe your typical morning routine (1-2 minutes)",
                "Talk about your favorite hobby or activity"
            ]
        }
    }
    
    return guide

def create_streamlit_recording_interface():
    """Create Streamlit interface for guided recording"""
    
    st.title("🎤 Clinical Voice Recording Protocol")
    st.markdown("### For Parkinson's Disease Detection")
    
    guide = create_recording_guide()
    
    # Protocol Overview
    st.header("📋 Recording Protocol")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Primary Task: Sustained Phonation")
        st.write("**Vowel Sounds to Record:**")
        for vowel in guide["clinical_protocol"]["vowel_sounds"]:
            st.write(f"• {vowel}")
        
        st.write(f"**Duration:** {guide['clinical_protocol']['duration']}")
        st.write(f"**Repetitions:** {guide['clinical_protocol']['repetitions']}")
        st.write(f"**Total:** {guide['clinical_protocol']['total_recordings']}")
    
    with col2:
        st.subheader("⚙️ Technical Requirements")
        specs = guide["recording_specifications"]
        st.write(f"**Sample Rate:** {specs['sample_rate']}")
        st.write(f"**Bit Depth:** {specs['bit_depth']}")
        st.write(f"**Format:** {specs['format']}")
        st.write(f"**Distance:** {specs['microphone_distance']}")
        st.write(f"**Environment:** {specs['environment']}")
    
    # Step-by-step instructions
    st.header("📝 Step-by-Step Instructions")
    
    with st.expander("🔧 Preparation"):
        for instruction in guide["clinical_instructions"]["preparation"]:
            st.write(f"• {instruction}")
    
    with st.expander("🎵 Recording Execution"):
        for instruction in guide["clinical_instructions"]["execution"]:
            st.write(f"• {instruction}")
    
    with st.expander("⏸️ Between Recordings"):
        for instruction in guide["clinical_instructions"]["between_recordings"]:
            st.write(f"• {instruction}")
    
    # Recording sequence
    st.header("🎬 Recording Sequence")
    
    if st.button("Start Guided Recording Session"):
        st.success("🎤 Recording session started!")
        
        vowels = [("Ahhhh", "/a/"), ("Ohhh", "/o/"), ("Ehhh", "/e/")]
        
        for i, (vowel_name, phonetic) in enumerate(vowels, 1):
            st.subheader(f"Vowel {i}: {vowel_name} {phonetic}")
            
            for rep in range(1, 4):
                st.write(f"**Recording {rep} of 3**")
                
                # Preparation countdown
                if st.button(f"Ready for {vowel_name} - Recording {rep}"):
                    with st.spinner("Get ready... 3"):
                        time.sleep(1)
                    with st.spinner("Get ready... 2"):
                        time.sleep(1)
                    with st.spinner("Get ready... 1"):
                        time.sleep(1)
                    
                    st.success(f"🔴 RECORD NOW: Say '{vowel_name}' steadily until you run out of breath")
                    st.info("💡 Tip: Keep the sound steady and consistent")
                
                st.write("---")
    
    # Additional tasks
    st.header("🔄 Additional Voice Tasks")
    
    with st.expander("🗣️ Diadochokinetic Tasks (Optional)"):
        st.write("These test rapid speech movements:")
        for task in guide["additional_tasks"]["diadochokinetic"]:
            st.write(f"• {task}")
    
    with st.expander("📖 Reading Tasks (Optional)"):
        for task in guide["additional_tasks"]["reading_tasks"]:
            st.write(f"• {task}")
        
        st.subheader("The Rainbow Passage")
        st.text("""
When the sunlight strikes raindrops in the air, they act as a prism and form a rainbow.
The rainbow is a division of white light into many beautiful colors. These take the
shape of a long round arch, with its path high above, and its two ends apparently
beyond the horizon. There is, according to legend, a boiling pot of gold at one end.
People look, but no one ever finds it. When a man looks for something beyond his
reach, his friends say he is looking for the pot of gold at the end of the rainbow.
        """)
    
    with st.expander("💬 Spontaneous Speech (Optional)"):
        for task in guide["additional_tasks"]["spontaneous_speech"]:
            st.write(f"• {task}")
    
    # Quality checklist
    st.header("✅ Recording Quality Checklist")
    
    quality_checks = [
        "No background noise or interruptions",
        "Consistent volume throughout recording",
        "Clear, steady vowel sound",
        "No voice breaks or cracks",
        "Proper microphone distance maintained",
        "WAV format, not compressed",
        "At least 3 seconds duration"
    ]
    
    for check in quality_checks:
        st.checkbox(check)
    
    # Important notes
    st.header("⚠️ Important Notes")
    
    st.warning("""
    **Medical Disclaimer:**
    - This is for research/screening purposes only
    - Not a substitute for professional medical diagnosis
    - Consult a neurologist if you have motor symptoms
    - Voice changes can have many causes besides Parkinson's
    """)
    
    st.info("""
    **Why These Specific Sounds?**
    - Sustained vowels reveal voice stability issues
    - Jitter and shimmer are measurable in steady sounds
    - /a/ vowel is most commonly used in research
    - Multiple repetitions ensure reliability
    """)
    
    st.success("""
    **Research Basis:**
    - Based on Parkinson's Voice Initiative protocols
    - Used in clinical studies worldwide
    - Validated for detecting voice changes in Parkinson's
    - Standard protocol in speech pathology
    """)

def main():
    create_streamlit_recording_interface()

if __name__ == "__main__":
    main()
