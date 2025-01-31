from flask import Flask, Response, render_template, request, jsonify
import os
import uuid
import json
from PIL import Image
import face_recognition
from flask import send_from_directory
import cv2


from twilio.rest import Client

# Twilio Configuration (replace with your actual credentials)
TWILIO_ACCOUNT_SID = "AC0f292488a1b530db5a7b74256caf50a5"
TWILIO_AUTH_TOKEN = "fb8a8d2005b1914f1bf3c84cf1754bca"
TWILIO_PHONE_NUMBER = "9110599762"



# Initialize Flask app
#app = Flask(__name__)

app = Flask(__name__, static_folder='data')

@app.route('/data/<path:filename>')
def serve_static(filename):
    return send_from_directory(app.static_folder, filename)


# Directories for storing data
DATA_DIR = "data"
IMAGES_DIR = os.path.join(DATA_DIR, "images")
VIDEOS_DIR = os.path.join(DATA_DIR, "videos")
MATCHED_FRAMES_DIR = os.path.join(DATA_DIR, "matched_frames")
DB_FILE = os.path.join(DATA_DIR, "missing_persons.json")


# Ensure directories exist
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(VIDEOS_DIR, exist_ok=True)
os.makedirs(MATCHED_FRAMES_DIR, exist_ok=True)

# Load or initialize the database
def load_database():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, 'r') as f:
            return json.load(f)
    return []

def save_database(db):
    with open(DB_FILE, 'w') as f:
        json.dump(db, f, indent=4)

# Load the existing database
database = load_database()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/add_missing_person", methods=["POST"])
def add_missing_person():
    try:
        # Get data from request
        name = request.form.get("name")
        age = request.form.get("age")
        description = request.form.get("description")
        mobile=request.form.get("mobile")
        image_file = request.files.get("image")

        if not (name and age and description and image_file):
            return jsonify({"error": "All fields (name, age,mobile, description, image) are required."}), 400

        # Save the image
        image_id = str(uuid.uuid4())
        image_path = os.path.join(IMAGES_DIR, f"{image_id}.jpg")
        image_file.save(image_path)

        # Process the image and extract face embeddings
        image = face_recognition.load_image_file(image_path)
        face_encodings = face_recognition.face_encodings(image)

        if len(face_encodings) == 0:
            os.remove(image_path)  # Clean up the image file if no face is detected
            return jsonify({"error": "No face detected in the image."}), 400

        # Use the first face encoding found
        face_encoding = face_encodings[0].tolist()

        # Create a new missing person record
        
        person = {
            "id": image_id,
            "name": name,
            "age": age,
            "description": description,
            "Mobie":mobile,
            "image_path": image_path,
            "face_encoding": face_encoding,
            "status": "missing",  # New status field
            "traced_location": None  # New location field
        }

        # Add to the database
        database.append(person)
        save_database(database)

        return jsonify({"message": "Missing person added successfully.", "person": person}), 201

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/get_missing_persons", methods=["GET"])
def get_missing_persons():
    try:
        return jsonify(database), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/match_person", methods=["POST"])
def match_person():
    try:
        image_file = request.files.get("image")
        traced_location = request.form.get("location")

        if not image_file:
            return jsonify({"error": "Image file is required."}), 400

        temp_image_path = os.path.join(IMAGES_DIR, "temp.jpg")
        image_file.save(temp_image_path)

        uploaded_image = face_recognition.load_image_file(temp_image_path)
        uploaded_encodings = face_recognition.face_encodings(uploaded_image)

        if len(uploaded_encodings) == 0:
            os.remove(temp_image_path)
            return jsonify({"error": "No face detected in the uploaded image."}), 400

        uploaded_encoding = uploaded_encodings[0]
        best_match = None
        matches_debug = []  # For debugging

        for person in database:
            stored_encoding = person["face_encoding"]
            distance = face_recognition.face_distance([stored_encoding], uploaded_encoding)[0]
            confidence = 1 - distance  # Confidence calculation
            matches_debug.append({
                "name": person["name"],
                "distance": distance,
                "confidence": confidence,
            })

            if distance <= 0.5 and (best_match is None or confidence > best_match["confidence"]):
                best_match = {
                    "id": person["id"],
                    "name": person["name"],
                    "age": person["age"],
                    "description": person["description"],
                    "image_path": person["image_path"],
                    "traced_location": traced_location,
                    "confidence": round(confidence * 100, 2),
                }

        os.remove(temp_image_path)

        # Log match details for debugging
        print("Match Debug Data:", matches_debug)

        if best_match:
            return jsonify({"message": "Match found!", **best_match}), 200

        return jsonify({"message": "No match found."}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route("/update_case_status", methods=["POST"])
def update_case_status():
    try:
        data = request.get_json()
        print(f"Received payload: {data}")  # Debug log
        person_id = data.get("id")
        traced_location = data.get("traced_location")

        if not person_id or not traced_location:
            print("Error: Missing required fields.")  # Debug log
            return jsonify({"error": "Missing required fields: id and traced_location"}), 400

        person = next((p for p in database if p["id"] == person_id), None)
        if person:
            person["status"] = "solved"
            person["traced_location"] = traced_location
            save_database(database)
            print(f"Case updated for ID: {person_id}")  # Debug log
            return jsonify({"message": "Case marked as solved.", "person": person}), 200
        else:
            print(f"Person with ID {person_id} not found.")  # Debug log
            return jsonify({"error": "Person not found."}), 404

    except Exception as e:
        print(f"Error: {str(e)}")  # Debug log
        return jsonify({"error": str(e)}), 500



@app.route("/get_solved_cases", methods=["GET"])
def get_solved_cases():
    try:
        solved_cases = [p for p in database if p["status"] == "solved"]
        return jsonify(solved_cases), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Serve images from the images directory
@app.route('/data/images/<filename>')
def serve_image(filename):
    return send_from_directory(IMAGES_DIR, filename)


# Initialize OpenCV tracker
def gen_frames():
    # Start the webcam capture
    cap = cv2.VideoCapture(0)  # 0 is typically the default webcam
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    while True:
        success, frame = cap.read()  # Read the current frame
        if not success:
            break

        # Convert the frame to RGB (OpenCV uses BGR by default)
        # rgb_frame = frame[:, :, ::-1]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Ensure that rgb_frame is not empty
        if rgb_frame is None or rgb_frame.size == 0:
            print("Error: Empty frame")
            continue

        # Find all face locations in the current frame
        face_locations = face_recognition.face_locations(rgb_frame)

        if len(face_locations) > 0:
            try:
                # Ensure face_locations is valid
                if any(loc[0] < 0 or loc[1] < 0 or loc[2] < 0 or loc[3] < 0 for loc in face_locations):
                    print("Error: Invalid face location coordinates")
                    continue
                
                # Find all face encodings for the detected faces
                face_encodings = face_recognition.face_encodings(rgb_frame)

            except Exception as e:
                print(f"Error while encoding faces: {e}")
                face_encodings = []  # In case of an error, just continue

            # Check and match the faces against the database
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                # Compare face with the database of missing persons
                matches = face_recognition.compare_faces([person['face_encoding'] for person in database], face_encoding, tolerance=0.6)

                matched = False  # Flag to check if a match is found
                for i, match in enumerate(matches):
                    if match:
                        matched = True
                        # Draw a box around the recognized face
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

                        # Label the person with their name (if found in the database)
                        name = database[i]['name']
                        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # If no match was found, no action is taken for this face
                if not matched:
                    continue  # No drawing or labeling if no match

        # Encode the frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            print("Error: Failed to encode frame")
            break

        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

    cap.release()  # Ensure the capture is released after the loop


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


#video processing
@app.route("/matchvideo_person", methods=["POST"])
def match_video_person():
    try:
        video_file = request.files.get("video")
        traced_location = request.form.get("location")

        if not video_file:
            return jsonify({"error": "Video file is required."}), 400

        video_id = str(uuid.uuid4())
        video_path = os.path.join(VIDEOS_DIR, f"{video_id}.mp4")
        video_file.save(video_path)
        print(f"Video saved to: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return jsonify({"error": "Unable to open video."}), 500

        all_database_encodings = [person["face_encoding"] for person in database]
        max_match = None
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = 0
        processed_frames = 0

        print("Starting optimized video processing...")

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            frame_count += 1
            if frame_count % 10 != 0:  # Process every 10th frame
                continue

            resized_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)  # Downscale frame
            rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

            face_locations = face_recognition.face_locations(rgb_frame, model="face_recognition_model.h5")
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            processed_frames += 1

            for face_encoding in face_encodings:
                distances = face_recognition.face_distance(all_database_encodings, face_encoding)
                min_distance = min(distances)
                confidence = 1 - min_distance

                if min_distance <= 0.6 and confidence > 0.5:  # Only consider matches above 50% confidence
                    matched_index = distances.argmin()
                    matched_person = database[matched_index]

                    match_time = frame_count / fps
                    matched_frame_path = os.path.join(MATCHED_FRAMES_DIR, f"{video_id}_frame_{frame_count}.jpg")
                    cv2.imwrite(matched_frame_path, frame)  # Save the matched frame

                    max_match = {
                        "message": "Match found in video!",
                        "name": matched_person["name"],
                        "age": matched_person["age"],
                        "description": matched_person["description"],
                        "image_path": matched_person["image_path"],
                        "matched_frame_path": f"/data/matched_frames/{os.path.basename(matched_frame_path)}",
                        "frame": frame_count,
                        "time": round(match_time, 2),
                        "confidence": round(confidence * 100, 2),
                    }
                    print(f"Match found: {max_match}")  # Log the match
                    break  # Stop further processing after finding a match

        cap.release()

        print(f"Total frames processed: {processed_frames}")
        if max_match:
            return jsonify(max_match), 200

        return jsonify({"message": "No match found in video."}), 200

    except Exception as e:
        print(f"Error during video processing: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/send_matched_image', methods=['POST'])
def send_matched_image():
    try:
        data = request.get_json()
        image_path = data.get("image_path")
        name = data.get("name")

        if not image_path or not name:
            return jsonify({"error": "Image path and name are required."}), 400

        # Construct the public URL for the image
        public_url = f"https://4aad-103-175-246-187.ngrok-free.app{image_path}"

        # Debug: Log the URL
        print(f"Media URL being sent: {public_url}")

        # Send the image via Twilio
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        message = client.messages.create(
            body=f"Match found for {name}. See the attached image.",
            from_=TWILIO_PHONE_NUMBER,
            to="+recipient_phone_number",  # Replace dynamically if needed
            media_url=[public_url]
        )

        print(f"SMS sent with SID: {message.sid}")
        return jsonify({"message": "Image sent successfully!"}), 200

    except Exception as e:
        print(f"Error sending SMS: {e}")
        return jsonify({"error": str(e)}), 500





if __name__ == "__main__":
    app.run(debug=True)