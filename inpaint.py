import sys
import math
import time
import cv2
import numpy as np
from line_profiler import profile


class Inpainter():
    # Inpainter settings
    DEFAULT_HALF_PATCH_WIDTH = 4
    MODE_ADDITION = 0
    MODE_MULTIPLICATION = 1

    # Input validation
    ERROR_INPUT_MAT_INVALID_TYPE = 0
    ERROR_INPUT_MASK_INVALID_TYPE = 1
    ERROR_MASK_INPUT_SIZE_MISMATCH = 2
    ERROR_HALF_PATCH_WIDTH_ZERO = 3
    CHECK_VALID = 4

    # Inpainter variables
    inputImage = None
    mask = None
    result = None
    image = None
    sourceRegion = None
    targetRegion = None
    originalSourceRegion = None
    gradientX = None
    gradientY = None
    confidence = None
    data = None
    LAPLACIAN_KERNEL = NORMAL_KERNELX = NORMAL_KERNELY = None

    # cv::Point2i
    bestMatchUpperLeft = bestMatchLowerRight = None
    patchHeight = patchWidth = 0

    # std::vector<cv::Point> -> list[(y,x)]
    fillFront = []

    # std::vector<cv::Point2f>
    normals = []
    sourcePatchULList = []
    targetPatchSList = []
    targetPatchTList = []
    halfPatchWidth = None
    targetIndex = None

    @profile
    def __init__(self, inputImage, mask, halfPatchWidth=DEFAULT_HALF_PATCH_WIDTH):
        self.inputImage = np.copy(inputImage)
        self.mask = np.copy(mask)
        self.mask = np.copy(mask)
        self.image = np.copy(inputImage)
        self.result = np.ndarray(
            shape=inputImage.shape, dtype=inputImage.dtype)
        self.halfPatchWidth = halfPatchWidth

    @profile
    def checkValidInputs(self):
        if not self.inputImage.dtype == np.uint8:  # CV_8UC3
            return self.ERROR_INPUT_MAT_INVALID_TYPE
        if not self.mask.dtype == np.uint8:  # CV_8UC1
            return self.ERROR_INPUT_MASK_INVALID_TYPE
        if not self.mask.shape == self.inputImage.shape[:2]:  # CV_ARE_SIZES_EQ
            return self.ERROR_MASK_INPUT_SIZE_MISMATCH
        if self.halfPatchWidth == 0:
            return self.ERROR_HALF_PATCH_WIDTH_ZERO
        return self.CHECK_VALID

    @profile
    def inpaint(self):
        self.initializeMats()
        self.calculateGradients()

        while True:
            self.computeFillFront()
            self.computeConfidence()
            self.computeData()
            self.computeTarget()

            print('Computing bestpatch', time.asctime())
            self.computeBestPatch()
            self.updateMats()

            # cv2.imwrite("updatedMask.jpg", self.updatedMask)
            cv2.imwrite("workImage.jpg", self.image)

            if self.checkEnd():
                break

        self.result = np.copy(self.image)

    @profile
    def initializeMats(self):
        _, self.confidence = cv2.threshold(
            self.mask, 10, 255, cv2.THRESH_BINARY)
        _, self.confidence = cv2.threshold(
            self.confidence, 2, 1, cv2.THRESH_BINARY_INV)

        self.sourceRegion = np.uint8(np.copy(self.confidence))
        self.originalSourceRegion = np.copy(self.sourceRegion)

        self.confidence = np.float32(self.confidence)

        _, self.targetRegion = cv2.threshold(
            self.mask, 10, 255, cv2.THRESH_BINARY)
        _, self.targetRegion = cv2.threshold(
            self.targetRegion, 2, 1, cv2.THRESH_BINARY)
        self.targetRegion = np.uint8(self.targetRegion)

        self.data = np.ndarray(
            shape=self.inputImage.shape[:2],  dtype=np.float32)

        # Initialise kernels
        self.LAPLACIAN_KERNEL = np.array(
            [[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float32)
        self.NORMAL_KERNELX = np.array(
            [[0, 0, 0], [-1, 0, 1], [0, 0, 0]], dtype=np.float32)
        self.NORMAL_KERNELY = self.NORMAL_KERNELX.T

    @profile
    def calculateGradients(self):
        srcGray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        self.gradientX = cv2.Scharr(srcGray, cv2.CV_32F, 1, 0)
        self.gradientX = cv2.convertScaleAbs(self.gradientX)
        self.gradientX = np.float32(self.gradientX)

        self.gradientY = cv2.Scharr(srcGray, cv2.CV_32F, 0, 1)
        self.gradientY = cv2.convertScaleAbs(self.gradientY)
        self.gradientY = np.float32(self.gradientY)

        self.gradientX[self.sourceRegion == 0] = 0
        self.gradientY[self.sourceRegion == 0] = 0

        self.gradientX /= 255
        self.gradientY /= 255

    @profile
    def computeFillFront(self):
        # Apply a Laplacian filter to find the boundary in the target region
        boundaryMat = cv2.filter2D(
            self.targetRegion, cv2.CV_32F, self.LAPLACIAN_KERNEL)

        # Apply normal kernels to the source region to find gradients
        sourceGradientX = cv2.filter2D(
            self.sourceRegion, cv2.CV_32F, self.NORMAL_KERNELX)
        sourceGradientY = cv2.filter2D(
            self.sourceRegion, cv2.CV_32F, self.NORMAL_KERNELY)

        # Find coordinates where boundaryMat is greater than zero, i.e. the contour
        y_indices, x_indices = np.where(boundaryMat > 0)

        # Extract the gradient values at these coordinates
        dx = sourceGradientX[y_indices, x_indices]
        dy = sourceGradientY[y_indices, x_indices]

        # Calculate normals and norms of normals
        normalX = dy
        normalY = -dx
        magnitude = np.sqrt(normalX**2 + normalY**2)

        # Normalize the normals
        nonzero = magnitude > 0
        normalX[nonzero] /= magnitude[nonzero]
        normalY[nonzero] /= magnitude[nonzero]

        # Store the coordinates of the fill front and the calculated normals
        self.fillFront = list(zip(x_indices, y_indices))
        self.normals = list(zip(normalX, normalY))

    @profile
    def getPatch(self, point):
        centerX, centerY = point
        height, width = self.image.shape[:2]

        minX = max(centerX - self.halfPatchWidth, 0)
        minY = max(centerY - self.halfPatchWidth, 0)

        maxX = min(centerX + self.halfPatchWidth, width - 1)
        maxY = min(centerY + self.halfPatchWidth, height - 1)

        upperLeft = (minX, minY)
        lowerRight = (maxX, maxY)

        return upperLeft, lowerRight

    @profile
    def computeConfidence(self):
        for pX, pY in self.fillFront:
            (aX, aY), (bX, bY) = self.getPatch((pX, pY))

            # Extract the patch from the targetRegion and confidence arrays
            patch_targetRegion = self.targetRegion[aY:bY+1, aX:bX+1]
            patch_confidence = self.confidence[aY:bY+1, aX:bX+1]

            # Compute the sum of confidence values where the targetRegion is 0 (non-target areas)
            total = np.sum(patch_confidence[patch_targetRegion == 0])

            # Calculate the total number of pixels in the patch
            total_pixels = (bX - aX + 1) * (bY - aY + 1)

            # Update the confidence at point (pX, pY) based on the total number of pixels in the patch
            self.confidence[pY, pX] = total / total_pixels

    @profile
    def computeData(self):
        for (x, y), (currentNormalX, currentNormalY) in zip(self.fillFront, self.normals):
            self.data[y, x] = abs(
                self.gradientX[y, x] * currentNormalX + self.gradientY[y, x] * currentNormalY) + 0.001

    @profile
    def computeTarget(self):
        omega, alpha, beta = 0.7, 0.2, 0.8
        self.targetIndex = 0

        # Assuming fillFront is a list of tuples, we convert relevant parts of confidence and data to arrays
        fillfront_indices = np.array(self.fillFront)

        indices = fillfront_indices[:, 1], fillfront_indices[:, 0]

        # Vectorized calculation of Rcp and priority for all points in fillFront
        Rcp = (1 - omega) * self.confidence[indices] + omega
        priorities = alpha * Rcp + beta * self.data[indices]

        # Way 2
        # priorities = self.data[indices] * self.confidence[indices]

        # Find the index of the maximum priority
        self.targetIndex = np.argmax(priorities)

    @profile
    def computeBestPatch(self):
        currentPoint = self.fillFront[self.targetIndex]
        (aX, aY), (bX, bY) = self.getPatch(currentPoint)
        pHeight, pWidth = bY - aY + 1, bX - aX + 1

        if pHeight != self.patchHeight or pWidth != self.patchWidth:
            self.patchHeight, self.patchWidth = pHeight, pWidth

            SUM_KERNEL = np.ones((pHeight, pWidth), dtype=np.uint8)
            convolvedMat = cv2.filter2D(
                self.originalSourceRegion, cv2.CV_8U, SUM_KERNEL, anchor=(0, 0))

            # sourcePatchULList: list whose elements is possible to be the UpperLeft of an patch to reference.
            area = pHeight * pWidth
            self.sourcePatchULList = np.argwhere(
                convolvedMat[:-pHeight, :-pWidth] == area)

        sourceRegion = self.sourceRegion[aY:aY + pHeight + 1,
                                         aX:aX + pWidth + 1]

        source_pixels = sourceRegion == 1

        self.targetPatchSList = np.argwhere(source_pixels)
        self.targetPatchTList = np.argwhere(~source_pixels)

        # Pre-compute values
        countedNum = float(len(self.targetPatchSList))
        minError = bestPatchVariance = sys.maxsize
        alpha, beta = 0.9, 0.5
        image_float32 = np.float32(self.image)

        for (y, x) in self.sourcePatchULList:

            i_indices = self.targetPatchSList[:, 0]
            j_indices = self.targetPatchSList[:, 1]

            source_pixels = image_float32[i_indices + y, j_indices + x]
            target_pixels = image_float32[i_indices + aY, j_indices + aX]

            differences = source_pixels - target_pixels

            patchError = np.sum(differences ** 2) / countedNum

            if alpha * patchError <= minError:
                i_indices = self.targetPatchTList[:, 0]
                j_indices = self.targetPatchTList[:, 1]
                sourcePixels = self.image[i_indices + y, j_indices + x, :]

                meanRGB = np.mean(sourcePixels, axis=0)
                difference = sourcePixels - meanRGB
                patchVariance = np.sum(difference ** 2)

                # Use alpha & Beta to encourage path with less patch variance.
                # For situations in which you need little variance.
                # Alpha = Beta = 1 to disable.
                if patchError < alpha * minError or patchVariance < beta * bestPatchVariance:
                    bestPatchVariance = patchVariance
                    minError = patchError
                    self.bestMatchUpperLeft = (x, y)
                    self.bestMatchLowerRight = (
                        x + pWidth - 1, y + pHeight - 1)

    @profile
    def updateMats(self):
        targetPoint = self.fillFront[self.targetIndex]
        tX, tY = targetPoint
        (aX, aY), _ = self.getPatch(targetPoint)
        bulX, bulY = self.bestMatchUpperLeft

        # Convert list of tuples to a NumPy array and get i and j arrays
        indices = np.array(self.targetPatchTList)
        i_array, j_array = indices[:, 0], indices[:, 1]

        # Compute coordinates of offset source and target indices
        source_indices = (bulY + i_array, bulX + j_array)
        target_indices = (aY + i_array, aX + j_array)

        # Update workImage, gradients, and confidence
        self.image[target_indices] = self.image[source_indices]
        self.gradientX[target_indices] = self.gradientX[source_indices]
        self.gradientY[target_indices] = self.gradientY[source_indices]
        self.confidence[target_indices] = self.confidence[tY, tX]

        # Update source and target regions, and the mask
        self.sourceRegion[target_indices] = 1
        self.targetRegion[target_indices] = 0
        self.mask[target_indices] = 0

    @profile
    def checkEnd(self):
        return np.all(self.sourceRegion != 0)


@profile
def inpaint(image, mask, patch_width=Inpainter.DEFAULT_HALF_PATCH_WIDTH):
    i = Inpainter(image, mask, patch_width)
    i.inpaint()
    return i.result