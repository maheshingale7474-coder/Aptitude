import streamlit as st
import pandas as pd
import random
import google.generativeai as genai

# Configure Gemini API
# It's recommended to use Streamlit secrets for API keys
# In your Streamlit app, create a .streamlit/secrets.toml file with:
# GOOGLE_API_KEY = "your_api_key_here"

# Check if secrets are configured, otherwise inform the user
if "GOOGLE_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    gemini_model = genai.GenerativeModel('gemini-pro')
else:
    st.error("API Key not found. Please set GOOGLE_API_KEY in Streamlit secrets.")
    st.stop()

# Load the dataset
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('aptitude_dataset.csv')
        return df
    except FileNotFoundError:
        st.error("aptitude_dataset.csv not found. Please ensure it's in the same directory.")
        st.stop()

df = load_data()

st.title("AI-Powered Adaptive Aptitude Learning System")

# Initialize session state for the test
if 'test_started' not in st.session_state:
    st.session_state.test_started = False
    st.session_state.num_questions = 0
    st.session_state.sample_questions = pd.DataFrame()
    st.session_state.current_question_index = 0
    st.session_state.score = 0
    st.session_state.performance_tracker = {
        'correct_by_topic': {}, 'incorrect_by_topic': {},
        'correct_by_difficulty': {}, 'incorrect_by_difficulty': {}
    }
    st.session_state.user_answers = []
    st.session_state.explanations = []


def start_test():
    st.session_state.test_started = True
    st.session_state.num_questions = st.session_state.selected_num_questions
    # Sample questions without random_state for variety
    st.session_state.sample_questions = df.sample(st.session_state.num_questions).reset_index(drop=True)
    st.session_state.current_question_index = 0
    st.session_state.score = 0
    st.session_state.performance_tracker = {
        'correct_by_topic': {}, 'incorrect_by_topic': {},
        'correct_by_difficulty': {}, 'incorrect_by_difficulty': {}
    }
    st.session_state.user_answers = []
    st.session_state.explanations = []

def submit_answer(selected_option):
    current_q_data = st.session_state.sample_questions.iloc[st.session_state.current_question_index]
    user_answer_char = None
    options_map = {
        current_q_data['option_a']: 'A',
        current_q_data['option_b']: 'B',
        current_q_data['option_c']: 'C',
        current_q_data['option_d']: 'D'
    }
    for opt_val, opt_char in options_map.items():
        if opt_val == selected_option:
            user_answer_char = opt_char
            break

    st.session_state.user_answers.append(user_answer_char)

    topic = current_q_data['topic']
    difficulty = current_q_data['difficulty']

    if user_answer_char == current_q_data['correct_answer']:
        st.session_state.score += 1
        st.session_state.performance_tracker['correct_by_topic'][topic] = st.session_state.performance_tracker['correct_by_topic'].get(topic, 0) + 1
        st.session_state.performance_tracker['correct_by_difficulty'][difficulty] = st.session_state.performance_tracker['correct_by_difficulty'].get(difficulty, 0) + 1
        st.session_state.explanations.append("Correct!")
    else:
        st.session_state.performance_tracker['incorrect_by_topic'][topic] = st.session_state.performance_tracker['incorrect_by_topic'].get(topic, 0) + 1
        st.session_state.performance_tracker['incorrect_by_difficulty'][difficulty] = st.session_state.performance_tracker['incorrect_by_difficulty'].get(difficulty, 0) + 1

        explanation_text = f"Incorrect. The correct answer was {current_q_data['correct_answer']}.\nExplanation: {current_q_data['explanation']}"

        # Generate more detailed explanation using LLM
        try:
            llm_prompt = f"""The following is an aptitude test question. The user answered incorrectly. Provide a more detailed explanation for why the correct answer is right and why the user's choice was wrong, based on the provided correct answer and existing explanation.

Question: {current_q_data['question']}
Options:
A) {current_q_data['option_a']}
B) {current_q_data['option_b']}
C) {current_q_data['option_c']}
D) {current_q_data['option_d']}
User's Answer: {user_answer_char}
Correct Answer: {current_q_data['correct_answer']}
Existing Explanation: {current_q_data['explanation']}

Provide a comprehensive step-by-step explanation for the correct answer, and briefly explain why the incorrect options (especially the user's choice) might be misleading."""

            llm_response = gemini_model.generate_content(llm_prompt)
            explanation_text += f"\n\n--- LLM-Generated Detailed Explanation ---\n" + llm_response.text + "\n-----------------------------------------"
        except Exception as e:
            explanation_text += f"\nError generating LLM explanation: {e}\nFalling back to static explanation."
        st.session_state.explanations.append(explanation_text)

    st.session_state.current_question_index += 1


if not st.session_state.test_started:
    st.write("Welcome to the AI-Powered Adaptive Aptitude Learning System!")
    st.session_state.selected_num_questions = st.number_input("How many questions would you like to attempt?", min_value=1, max_value=len(df), value=5)
    st.button("Start Test", on_click=start_test)
else:
    if st.session_state.current_question_index < st.session_state.num_questions:
        current_q_data = st.session_state.sample_questions.iloc[st.session_state.current_question_index]

        st.subheader(f"Question {st.session_state.current_question_index + 1}/{st.session_state.num_questions}")
        st.write(current_q_data['question'])

        options = [
            current_q_data['option_a'],
            current_q_data['option_b'],
            current_q_data['option_c'],
            current_q_data['option_d']
        ]
        selected_option = st.radio("Choose your answer:", options, key=f"question_{st.session_state.current_question_index}")

        if st.button("Submit Answer", key=f"submit_{st.session_state.current_question_index}"):
            if selected_option:
                submit_answer(selected_option)
                st.rerun() # Rerun to show next question or results
            else:
                st.warning("Please select an option before submitting.")

    else:
        st.subheader("Test Completed!")
        st.write(f"You scored {st.session_state.score} out of {st.session_state.num_questions}.")

        st.subheader("--- Performance Summary ---")
        st.write("**Correct by Topic:**", st.session_state.performance_tracker['correct_by_topic'])
        st.write("**Incorrect by Topic:**", st.session_state.performance_tracker['incorrect_by_topic'])
        st.write("**Correct by Difficulty:**", st.session_state.performance_tracker['correct_by_difficulty'])
        st.write("**Incorrect by Difficulty:**", st.session_state.performance_tracker['incorrect_by_difficulty'])

        if st.session_state.performance_tracker['incorrect_by_topic']:
            most_incorrect_topic = max(st.session_state.performance_tracker['incorrect_by_topic'], key=st.session_state.performance_tracker['incorrect_by_topic'].get)
            st.info(f"Suggestion: You seem to struggle most with '{most_incorrect_topic}' questions. Focus on this topic for improvement.")

        if st.session_state.performance_tracker['incorrect_by_difficulty']:
            most_incorrect_difficulty = max(st.session_state.performance_tracker['incorrect_by_difficulty'], key=st.session_state.performance_tracker['incorrect_by_difficulty'].get)
            st.info(f"Suggestion: You had difficulty with '{most_incorrect_difficulty}' questions. Consider reviewing your basics or practicing more questions of this difficulty.")

        st.markdown("--- Detailed Review ---")
        for i, (q_data, user_ans_char, explanation) in enumerate(zip(st.session_state.sample_questions.iterrows(), st.session_state.user_answers, st.session_state.explanations)):
            idx, row = q_data
            st.markdown(f"**Question {i+1}:** {row['question']}")
            st.write(f"Your Answer: {user_ans_char}")
            st.write(f"Correct Answer: {row['correct_answer']}")
            st.markdown(f"**Explanation:**\n{explanation}")
            st.markdown("\n---")

        if st.button("Retake Test"):
            st.session_state.test_started = False
            st.experimental_rerun()
