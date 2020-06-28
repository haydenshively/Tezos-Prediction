"""
RESULTS
Reads results from file and displays them
"""

if __name__ == '__main__':
    from numpy import load, hstack
    from time import sleep
    import cv2

    truths = load('Results/truths.npy')
    predictions = load('Results/predictions.npy')

    for truth, prediction in zip(truths, predictions):
        cv2.imshow('Results', hstack((truth, prediction)))
        ch = 0xFF & cv2.waitKey(1)
        if ch == 27: break

        sleep(0.5)

    cv2.destroyAllWindows()
