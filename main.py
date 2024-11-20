# Install required libraries

import gradio as gr
import numpy as np
import requests
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load pre-trained MobileNetV2 model
base_model = MobileNetV2(weights="imagenet", include_top=False, pooling="avg")

# Fetch weather data using Open-Meteo API
def get_weather(latitude=0, longitude=0):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current_weather=true"
    response = requests.get(url).json()
    if "current_weather" in response:
        temperature = response["current_weather"]["temperature"]
        windspeed = response["current_weather"]["windspeed"]
        weather = response["current_weather"]["weathercode"]
        return temperature, windspeed, weather
    return None, None, "Weather data unavailable"

# Analyze crop image
def analyze_crop(image, latitude, longitude):
    # Preprocess the image
    image = np.array(image.resize((224, 224))) / 255.0
    image_array = np.expand_dims(image, axis=0)

    # Extract features from the image using the pre-trained model
    features = base_model.predict(image_array).flatten().reshape(1, -1)

    # Simulated clustering logic
    kmeans = KMeans(n_clusters=5, random_state=42)
    kmeans.fit(np.random.rand(100, features.shape[1]))  # Dummy data for simulation
    cluster = kmeans.predict(features)[0]

    # Irrigation categories
    irrigation_report = {
        0: "Healthy: No irrigation needed",
        1: "Slightly Dry: Irrigation needed in 2-3 days",
        2: "Dry: Irrigate in 1-2 days",
        3: "Overwatered: Delay irrigation for 3-4 days",
        4: "Critical: Immediate irrigation needed",
    }

    irrigation_days = {
        0: "No irrigation needed for the next 2-3 days.",
        1: "Irrigate lightly in the next 2-3 days.",
        2: "Irrigate in the next 1-2 days.",
        3: "Hold off on irrigation for 3-4 days.",
        4: "Irrigate immediately or risk crop damage.",
    }

    # Get weather data
    temperature, windspeed, weather = get_weather(latitude, longitude)

    # Interpret weather code (dummy example, expand as needed)
    weather_description = {
        0: "Clear sky",
        1: "Mainly clear",
        2: "Partly cloudy",
        3: "Overcast",
        45: "Fog",
        48: "Depositing rime fog",
        51: "Light drizzle",
        53: "Moderate drizzle",
        55: "Dense drizzle",
        61: "Light rain",
        63: "Moderate rain",
        65: "Heavy rain",
        71: "Light snow",
        73: "Moderate snow",
        75: "Heavy snow",
        95: "Thunderstorm",
    }.get(weather, "Unknown weather condition")

    # Generate enhanced recommendations
    additional_note = "Normal conditions."
    if weather in {61, 63, 65}:
        additional_note = "Rain is expected. Adjust irrigation accordingly."
    elif temperature > 35:
        additional_note = "High temperature detected. Consider earlier irrigation."
    elif windspeed > 20:
        additional_note = "Strong winds detected. Monitor soil moisture closely."

    # Generate output
    report = irrigation_report[cluster]
    suggestion = irrigation_days[cluster]
    weather_info = f"Weather: {weather_description}, Temperature: {temperature}°C, Wind Speed: {windspeed} km/h.\n{additional_note}"

    # Generate visual report
    plt.figure(figsize=(6, 4))
    categories = ["Healthy", "Slightly Dry", "Dry", "Overwatered", "Critical"]
    values = [30, 20, 25, 15, 10]  # Simulated values
    plt.bar(categories, values, color=["green", "yellow", "orange", "blue", "red"])
    plt.title("Crop Health Distribution")
    plt.xlabel("Category")
    plt.ylabel("Percentage")
    plt.savefig("crop_report.png")

    return report, suggestion, weather_info, "crop_report.png"

# Gradio interface
interface = gr.Interface(
    fn=analyze_crop,
    inputs=[
        gr.inputs.Image(type="pil", label="Upload Crop Image"),
        gr.inputs.Number(default=8.5241, label="Enter Latitude (e.g., 8.5241 for Tvm)"),
        gr.inputs.Number(default=76.9366, label="Enter Longitude (e.g., 76.9366 for Tvm)"),
    ],
    outputs=[
        gr.outputs.Textbox(label="Irrigation Report"),
        gr.outputs.Textbox(label="Suggested Irrigation Time"),
        gr.outputs.Textbox(label="Weather Information"),
        gr.outputs.Image(type="file", label="Health Distribution"),
    ],
    title="Advanced Crop Irrigation Assistant for Thiruvananthapuram",
    description="Upload a crop image and use Thiruvananthapuram's default latitude and longitude for a weather-adjusted irrigation report.",
)

# Launch the interface
interface.launch()
