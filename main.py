from google import genai
from google.genai import types
import cv2
import base64
import json
import pyttsx3

client = genai.Client()


def getCameraFrame():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode('utf-8')

def detectPhone():
    response = client.models.generate_content(
        model="gemma-4-31b-it", 
        contents=[
            types.Part.from_bytes(
                data=getCameraFrame(),
                mime_type='image/jpeg',
            ),
        ],
       config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level="minimal"),
            response_mime_type="application/json",
            response_schema=genai.types.Schema(
                type = genai.types.Type.OBJECT,
                required = ["phone"],
                properties = {
                    "phone": genai.types.Schema(
                        type = genai.types.Type.BOOLEAN,
                    ),
                },
            ),
            system_instruction=[
                types.Part.from_text(text="""Analyze the photo of the user. Determine if they are holding a phone and looking at it. If they are, return True. Otherwise, return False."""),
            ],
        )
    )
    detectedPhone = response.text
    detectedPhone = json.loads(detectedPhone)
    return detectedPhone["phone"]

def main():
    phone_detect_count = 0
    while True:
        if detectPhone():
            print("Phone detected!!")
            phone_detect_count += 1
            if phone_detect_count >= 5:
                print("Phone detected 5 times!")
                pyttsx3.speak("Put the phone down and subscribe to Jeremy Quan!")
                break
        else:
            print("Phone not detected")
            phone_detect_count = 0

pyttsx3.speak("Put the phone down and subscribe to Jeremy Quan!")