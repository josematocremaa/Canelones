import cv2
import mediapipe as mp

def main():
    # Inicializar las utilidades de MediaPipe para manos y dibujo
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    # Iniciar la captura de video (0 suele ser la webcam por defecto)
    cap = cv2.VideoCapture(0)

    # Configurar el modelo de detección de manos
    with mp_hands.Hands(
        model_complexity=0,
        max_num_hands=4,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
        
        print("Iniciando cámara... Presiona 'ESC' para salir.")
        
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignorando frame vacío de la cámara.")
                continue

            # Para mejorar el rendimiento, marcamos la imagen como no escribible
            image.flags.writeable = False
            # MediaPipe usa RGB, OpenCV usa BGR. Hay que convertirla.
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Procesar la imagen para detectar las manos
            results = hands.process(image)

            # Volver a hacer la imagen escribible y pasarla a BGR para mostrarla en OpenCV
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Si se detectan manos, dibujar los puntos clave (landmarks)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Aquí puedes acceder a las coordenadas (x, y, z) de cada punto de la mano
                    # Ejemplo: dedo_indice = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())

            # Voltear la imagen horizontalmente para un efecto de espejo (más natural)
            image = cv2.flip(image, 1)
            
            # Mostrar la ventana
            cv2.imshow('Detector de Manos (Preparacion para ROS)', image)
            
            # Salir si se presiona la tecla ESC
            if cv2.waitKey(5) & 0xFF == 27:
                break

    # Liberar la cámara y cerrar ventanas
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
