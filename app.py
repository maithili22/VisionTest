import streamlit as st
from PIL import Image
import os
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI
from google.generativeai import GenerativeModel
import google.generativeai as genai

os.environ["GOOGLE_API_KEY"] = "your-api-key"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])


vision_model = GenerativeModel('gemini-1.5-flash')


def get_image_description(image):
    response = vision_model.generate_content(["Describe the UI elements and layout of this image in detail:", image])
    return response.text


def generate_testcase(description, context=None):
    example_test_cases = """
    Example 1:
    Test Case 1: Login Form Validation
    - Description: Test the validation of the login form fields.
    - Pre-conditions: The login page is open.
    - Testing Steps:
        1. Leave the username field empty.
        2. Enter an invalid email format in the email field.
        3. Enter a short password.
        4. Click the login button.
    - Expected Result: Error messages should appear for each invalid field.

    Example 2:
    Test Case 2: Navigation Menu Functionality
    - Description: Test the functionality of the navigation menu items.
    - Pre-conditions: The user is logged in and on the dashboard.
    - Testing Steps:
        1. Click each menu item.
        2. Verify that the correct page loads for each item.
        3. Check that the active menu item is highlighted.
    - Expected Result: Each menu item should lead to the correct page, and the active item should be visually distinct.
    """

    prompt_template = PromptTemplate(
        input_variables=["description", "context"],
        template=f"""
        {example_test_cases}

        Based on the following description of a UI screenshot: "{{description}}", 
        and the additional context '{{context}}', generate detailed test cases for testing the features visible in this UI.
        Each test case should include:
        - Description: What the test case is about.
        - Pre-conditions: What needs to be set up or ensured before testing.
        - Testing Steps: Clear, step-by-step instructions on how to perform the test.
        - Expected Result: What should happen if the feature works correctly.

        Focus on creating test cases for the main features and interactions visible in the UI.
        Be specific to the elements described, but also consider common UI patterns that might be present.

        The output should be in the format:
        Test Case 1: Heading
        - Description
        - Pre-conditions
        - Testing Steps
        - Expected Result

        Test Case 2: Heading
        ....
        """
    )

    llm = ChatGoogleGenerativeAI(model="gemini-pro")
    chain = LLMChain(llm=llm, prompt=prompt_template)
    result = chain.run(description=description, context=context if context else "No additional context provided")

    return result


def main():
    st.title("Test Case Generator")
    st.write("Upload one or more screenshots of any UI and optionally provide context to generate detailed test cases.")

    # Allow users to upload multiple images
    uploaded_files = st.file_uploader("Upload UI screenshots", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

    context = st.text_area("Optional context for the test cases")

    if st.button("Generate Test Cases"):
        if uploaded_files:
            for idx, uploaded_file in enumerate(uploaded_files):
                image = Image.open(uploaded_file)

                with st.spinner(f"Analyzing image {idx + 1}: {uploaded_file.name}..."):
                    image_description = get_image_description(image)

                # Display the description and test case in the specified format
                st.subheader(f"Image {idx + 1}: {uploaded_file.name}")
                

                with st.spinner(f"Generating test cases for image {idx + 1}..."):
                    test_cases = generate_testcase(image_description, context)

                st.write("**Test Case**:")
                st.write(test_cases)
        else:
            st.error("Please upload at least one screenshot.")

if __name__ == "__main__":
    main()
