import numpy as np
import cv2


class KalmanFilter:
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)


    def predict(self):
        ''' This function estimates the position of the object'''
        predicted = self.kf.predict()
        x, y = np.float32(predicted[0][0]), np.float32(predicted[1][0])
        return x, y
    

    def update(self, coordX, coordY):
        measured = np.array([[np.float32(coordX)], [np.float32(coordY)]])
        self.kf.correct(measured)


    def getNextStates(self, statesNum):
        predictions = []
        for _ in range(statesNum):
            # Predict the next state
            prediction = self.kf.predict()
            # Get the updated state estimate
            state = self.kf.statePost
            predictions.append(prediction[0][0])
            # Set the current state as the new initial state for the next prediction
            self.kf.statePre = state
        return predictions