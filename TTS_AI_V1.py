import pyttsx3
import speech_recognition as sr
from openai import OpenAI

# Initialize the OpenAI client
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Set properties for the text-to-speech engine
engine.setProperty('rate', 150)    # Speed percent (can go over 100)
engine.setProperty('volume', 0.9)  # Volume 0-1

# Initialize the recognizer
recognizer = sr.Recognizer()

# Initialize the conversation with a system message
messages = [
    {"role": "system", "content": "You possess a poetic, eloquent, and slightly theatrical way of speaking. You respond to questions with a mix of sophistication, wit, and a touch of disdain, always providing insightful and profound answers. Remember to maintain an air of mystery and charisma in your responses, often quoting literature, history, or philosophy. Your goal is to educate and enlighten, but never without a flourish of drama and a hint of condescension."}
    #{"role": "system", "content": "You are a helpful assistant who answers questions."}
    ]


while True:
    try:
        # Use the microphone as source for input
        with sr.Microphone() as source:
            print("Listening...")
            # Adjust the recognizer sensitivity to ambient noise
            recognizer.adjust_for_ambient_noise(source, duration=0.2)
            # Listen for the user's input
            audio = recognizer.listen(source)

            # Recognize speech using Google Web Speech API
            user_input = recognizer.recognize_google(audio)
            user_input = user_input.lower()
            print(f"You: {user_input}")

            # Append user input to the messages
            messages.append({"role": "user", "content": user_input})

            # Create a completion
            completion = client.chat.completions.create(
                model="model-identifier",
                messages=messages,
                temperature=0.7,
            )

            # Extract the response text from the completion
            response_text = completion.choices[0].message.content
            print(f"AI: {response_text}")

            # Add the response text to the speech engine
            engine.say(response_text)

            # Run the speech engine
            engine.runAndWait()

            # Append the AI response to the messages
            messages.append({"role": "assistant", "content": response_text})

    except sr.UnknownValueError:
        print("Sorry, I did not understand that.")
    except sr.RequestError as e:
        print(f"Could not request results; {e}")
