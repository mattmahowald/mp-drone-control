import numpy as np
import mediapipe as mp
from typing import List, Dict
from sklearn.metrics import accuracy_score, classification_report
from mp_drone_control.utils.logging_config import setup_logging
from mediapipe.framework.formats import landmark_pb2

logger = setup_logging()


class MediaPipeBaseline:
    """Baseline model using MediaPipe's built-in hand gesture recognition."""

    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_gesture = mp.solutions.hands.Hands(
            static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5
        )
        # Map MediaPipe gestures to our labels
        self.gesture_map = {
            "ZERO": 0,  # Closed fist
            "ONE": 1,  # Index finger up
            "TWO": 2,  # Index and middle fingers up
            "THREE": 3,  # Index, middle, and ring fingers up
            "FOUR": 4,  # All fingers up except thumb
            "FIVE": 5,  # All fingers up
            "SIX": 6,  # Thumb and pinky up
            "SEVEN": 7,  # Thumb, index, and middle fingers up
            "EIGHT": 8,  # Thumb and index finger up
            "NINE": 9,  # Index finger bent
        }

    def predict(self, landmarks: np.ndarray) -> int:
        """
        Predict gesture class from landmarks.

        Args:
            landmarks: Array of shape (21, 3) containing hand landmarks

        Returns:
            Predicted class index
        """
        # Convert landmarks to MediaPipe format
        mp_landmarks = []
        for i in range(21):
            x, y, z = landmarks[i]
            mp_landmarks.append(
                landmark_pb2.NormalizedLandmark(x=float(x), y=float(y), z=float(z))
            )

        # Get gesture prediction
        gesture = self._get_gesture_from_landmarks(mp_landmarks)
        return self.gesture_map.get(gesture, 0)  # Default to FIST if unknown

    def _get_gesture_from_landmarks(
        self, landmarks: List[mp.solutions.hands.HandLandmark]
    ) -> str:
        """
        Determine ASL digit from landmarks using relative finger positions.
        """
        # Get key points
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]
        wrist = landmarks[0]

        # Calculate if each finger is extended (y coordinate is above wrist)
        thumb_up = thumb_tip.y < wrist.y
        index_up = index_tip.y < wrist.y
        middle_up = middle_tip.y < wrist.y
        ring_up = ring_tip.y < wrist.y
        pinky_up = pinky_tip.y < wrist.y

        # Calculate distances between fingertips and wrist
        thumb_dist = np.linalg.norm([thumb_tip.x - wrist.x, thumb_tip.y - wrist.y])
        index_dist = np.linalg.norm([index_tip.x - wrist.x, index_tip.y - wrist.y])
        middle_dist = np.linalg.norm([middle_tip.x - wrist.x, middle_tip.y - wrist.y])
        ring_dist = np.linalg.norm([ring_tip.x - wrist.x, ring_tip.y - wrist.y])
        pinky_dist = np.linalg.norm([pinky_tip.x - wrist.x, pinky_tip.y - wrist.y])

        # Calculate if fingers are bent (close to palm)
        thumb_bent = thumb_dist < 0.2
        index_bent = index_dist < 0.2
        middle_bent = middle_dist < 0.2
        ring_bent = ring_dist < 0.2
        pinky_bent = pinky_dist < 0.2

        # ASL digit recognition rules
        if all([thumb_bent, index_bent, middle_bent, ring_bent, pinky_bent]):
            return "ZERO"  # All fingers bent (closed fist)
        elif index_up and not any([middle_up, ring_up, pinky_up]):
            if thumb_up:
                return "ONE"  # Index finger up, thumb up
            else:
                return "ONE"  # Just index finger up
        elif index_up and middle_up and not any([ring_up, pinky_up]):
            if thumb_up:
                return "TWO"  # Index and middle up, thumb up
            else:
                return "TWO"  # Just index and middle up
        elif index_up and middle_up and ring_up and not pinky_up:
            if thumb_up:
                return "THREE"  # Index, middle, ring up, thumb up
            else:
                return "THREE"  # Just index, middle, ring up
        elif all([index_up, middle_up, ring_up, pinky_up]) and not thumb_up:
            return "FOUR"  # All fingers up except thumb
        elif all([thumb_up, index_up, middle_up, ring_up, pinky_up]):
            return "FIVE"  # All fingers up
        elif thumb_up and pinky_up and not any([index_up, middle_up, ring_up]):
            return "SIX"  # Thumb and pinky up
        elif thumb_up and index_up and middle_up and not any([ring_up, pinky_up]):
            return "SEVEN"  # Thumb, index, middle up
        elif thumb_up and index_up and not any([middle_up, ring_up, pinky_up]):
            return "EIGHT"  # Thumb and index up
        elif index_up and not any([thumb_up, middle_up, ring_up, pinky_up]):
            # Check if index finger is bent (close to palm)
            if index_bent:
                return "NINE"  # Index finger bent
            return "ONE"  # If not bent, it's just one
        else:
            return "ZERO"  # Default to zero if no clear match

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evaluate baseline model on test set.

        Args:
            X_test: Array of shape (N, 21, 3) containing test landmarks
            y_test: Array of shape (N,) containing true labels

        Returns:
            Dictionary containing evaluation metrics
        """
        predictions = []
        for landmarks in X_test:
            # Reshape if input is flattened
            if landmarks.shape == (63,):
                landmarks = landmarks.reshape(21, 3)
            pred = self.predict(landmarks)
            predictions.append(pred)

        predictions = np.array(predictions)

        # Calculate metrics
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(
            y_test, predictions, output_dict=True, zero_division=0
        )

        results = {"accuracy": accuracy, "classification_report": report}

        logger.info(f"Baseline accuracy: {accuracy:.3f}")
        logger.info("\nClassification Report:")
        logger.info(classification_report(y_test, predictions, zero_division=0))

        return results
