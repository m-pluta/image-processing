import sys
import cv2
import numpy as np


class Inpainter():
    # Inpainter settings
    DEFAULT_HALF_PATCH_WIDTH = 3

    # Input validation
    ERROR_INPUT_MAT_INVALID_TYPE = 0
    ERROR_INPUT_MASK_INVALID_TYPE = 1
    ERROR_MASK_INPUT_SIZE_MISMATCH = 2
    ERROR_HALF_PATCH_WIDTH_ZERO = 3
    CHECK_VALID = 4

    # Inpainter variables
    image = None
    shape = None
    mask = None
    result = None
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
    targetPoint = None
    Point = tuple[int, int]

    def __init__(self, image: np.ndarray, mask: np.ndarray, halfPatchWidth: int = 4):
        """Constructor for the inpainter

        Args:
            image (np.ndarray): Image to be inpainted
            mask (np.ndarray): Boolean Mask for the inpaint region
            halfPatchWidth (int, optional): Half the width of a Patch. Defaults to 4.
        """
        self.image = np.copy(image)
        self.mask = np.copy(mask)
        self.halfPatchWidth = halfPatchWidth
        self.shape = self.image.shape[:2]

    def checkValidInputs(self):
        """Check if the inputs provided to the Inpainter are valid

        Returns:
            int: The corresponding validity flag depending on the inputs provided
        """
        if not self.image.dtype == np.uint8:  # CV_8UC3
            return self.ERROR_INPUT_MAT_INVALID_TYPE
        if not self.mask.dtype == bool:  # Boolean mask
            return self.ERROR_INPUT_MASK_INVALID_TYPE
        if not self.mask.shape == self.shape:  # CV_ARE_SIZES_EQ
            return self.ERROR_MASK_INPUT_SIZE_MISMATCH
        if self.halfPatchWidth == 0:
            return self.ERROR_HALF_PATCH_WIDTH_ZERO
        return self.CHECK_VALID

    def inpaint(self):
        """Main loop method for the inpainter 
        """
        # Initialise working matrices and gradients
        self.initializeMats()
        self.calculateGradients()

        while True:
            # Compute the 'fill front', which is the boundary of the region to be inpainted.
            self.computeFillFront()

            # Calculate the confidence values for each point on the fill front,
            # which influences where the patching should prioritize.
            self.computeConfidence()

            # Compute the data term, which guides the selection of the patch based on structure.
            self.computeData()

            # Determine the target patch that best matches the structure and texture outside the fill region.
            self.computeTarget()

            # Find the best patch from the source region to copy from.
            self.computeBestPatch()

            # Update matrices to reflect the changes after the current inpainting iteration.
            self.updateMats()

            # Check if the inpainting process should end (e.g., when all regions are filled).
            if self.checkEnd():
                break

        self.result = np.copy(self.image)

    def initializeMats(self):
        """Initialises all working matrices of the Inpainter
        """
        # Define regions
        self.targetRegion = np.uint8(self.mask.astype(int))
        self.originalSourceRegion = np.uint8((~self.mask).astype(int))
        self.sourceRegion = np.copy(self.originalSourceRegion)

        # Set confidence of each pixel
        self.confidence = np.float32(self.sourceRegion)

        # Declare empty data array
        self.data = np.ndarray(
            shape=self.image.shape[:2],  dtype=np.float32)

        # Initialise kernels
        self.LAPLACIAN_KERNEL = np.array(
            [[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float32)
        self.NORMAL_KERNELX = np.array(
            [[0, 0, 0], [-1, 0, 1], [0, 0, 0]], dtype=np.float32)
        self.NORMAL_KERNELY = self.NORMAL_KERNELX.T

    def calculateGradients(self):
        """
        Calculates normalized gradients of the image in both x and y directions using the Scharr operator.
        """
        # Convert image to grayscale for gradient calculation.
        srcGray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # Compute x-gradient, convert to absolute, float32, and normalize.
        self.gradientX = cv2.Scharr(srcGray, cv2.CV_32F, 1, 0)
        self.gradientX = cv2.convertScaleAbs(self.gradientX)
        self.gradientX = np.float32(self.gradientX)

        # Compute y-gradient, convert to absolute, float32, and normalize.
        self.gradientY = cv2.Scharr(srcGray, cv2.CV_32F, 0, 1)
        self.gradientY = cv2.convertScaleAbs(self.gradientY)
        self.gradientY = np.float32(self.gradientY)

        # Zero gradients outside the source region.
        self.gradientX[self.sourceRegion == 0] = 0
        self.gradientY[self.sourceRegion == 0] = 0

        # Normalise the remaining gradients
        self.gradientX /= 255
        self.gradientY /= 255

    def computeFillFront(self):
        """
        Computes the fill front for the inpainting process, which includes identifying boundary pixels,
        calculating their gradients, and normalizing the normal vectors to these boundaries.
        """
        # Detect boundary of the target region using a Laplacian kernel.
        boundaryMat = cv2.filter2D(
            self.targetRegion, cv2.CV_32F, self.LAPLACIAN_KERNEL)

        # Apply gradient filters to the source region to find gradient components.
        sourceGradientX = cv2.filter2D(
            self.sourceRegion, cv2.CV_32F, self.NORMAL_KERNELX)
        sourceGradientY = cv2.filter2D(
            self.sourceRegion, cv2.CV_32F, self.NORMAL_KERNELY)

        # Identify boundary coordinates by finding non-zero values in the boundary matrix.
        y_indices, x_indices = np.where(boundaryMat > 0)

        # Extract the gradient values at the boundary coordinates.
        dx = sourceGradientX[y_indices, x_indices]
        dy = sourceGradientY[y_indices, x_indices]

        # Compute the normals by swapping and negating gradient components.
        normalX = dy
        normalY = -dx

        # Calculate the magnitude of the normals.
        magnitude = np.sqrt(normalX**2 + normalY**2)

        # Normalize the normal vectors.
        nonzero = magnitude > 0
        normalX[nonzero] /= magnitude[nonzero]
        normalY[nonzero] /= magnitude[nonzero]

        # Store the fill front coordinates and their corresponding normalized normals.
        self.fillFront = list(zip(x_indices, y_indices))
        self.normals = list(zip(normalX, normalY))

    def getPatch(self, point: Point):
        """
        Calculates the upper left and lower right coordinates of a square patch centered around a given point.
        The size of the patch is determined by `halfPatchWidth`, ensuring that the patch remains within the image boundaries.

        Parameters:
        - point (Point): The center (x, y) of the patch.

        Returns:
        - tuple[Point, Point]: Contains two points representing the coordinates of the upper left and lower right corners of the patch.
        """
        # Extract center coordinates from the point.
        centerX, centerY = point

        # Retrieve dimensions of the image.
        height, width = self.shape

        # Calculate the upper left corner, ensuring it doesn't go outside the image boundaries.
        minX = max(centerX - self.halfPatchWidth, 0)
        minY = max(centerY - self.halfPatchWidth, 0)

        # Calculate the lower right corner, ensuring it stays within the image boundaries.
        maxX = min(centerX + self.halfPatchWidth, width - 1)
        maxY = min(centerY + self.halfPatchWidth, height - 1)

        # Define the upper left and lower right corners of the patch.
        upperLeft = (minX, minY)
        lowerRight = (maxX, maxY)

        return upperLeft, lowerRight

    def computeConfidence(self):
        """
        Updates confidence values for each point on the fill front based on the surrounding area's confidence.
        """
        for pX, pY in self.fillFront:
            # Get coordinates of the patch surrounding the current point.
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

    def computeData(self):
        """
        Computes the data term for each point on the fill front. The data term is based on the dot product
        of the gradient at the point and the normal to the fill front at that point, emphasizing areas with high
        edge information.
        """
        for (x, y), (currNormX, currNormY) in zip(self.fillFront, self.normals):
            # Calculate the data term: the absolute value of the dot product of gradient and normal,
            # small constant added for numerical stability.
            self.data[y, x] = abs(
                self.gradientX[y, x] * currNormX + self.gradientY[y, x] * currNormY) + 0.001

    def computeTarget(self):
        """
        Determines the target point on the fill front with the highest priority for inpainting.
        """
        # Constants for weighting the terms in the priority calculation
        omega, alpha, beta = 0.7, 0.2, 0.8

        # Transpose indices for array access.
        fillfront_indices = np.array(self.fillFront)
        indices = fillfront_indices[:, 1], fillfront_indices[:, 0]

        # Compute the modified confidence term, 'Rcp'.
        Rcp = (1 - omega) * self.confidence[indices] + omega
        # Calculate priorities by combining 'Rcp' with the data term, weighted by alpha and beta.
        priorities = alpha * Rcp + beta * self.data[indices]

        # Identify the index with the maximum priority value.
        targetIndex = np.argmax(priorities)

        # Set the target point for the next inpainting step.
        self.targetPoint = self.fillFront[targetIndex]

    def computeBestPatch(self):
        """
        Identifies the best patch in the source area to replace the target region in the inpainting process.
        This involves calculating the best matching patch based on error minimization and texture consistency.
        """
        currentPoint = self.targetPoint
        (aX, aY), (bX, bY) = self.getPatch(currentPoint)
        pHeight, pWidth = bY - aY + 1, bX - aX + 1

        # Adjust kernel size if patch dimensions change.
        if pHeight != self.patchHeight or pWidth != self.patchWidth:
            self.patchHeight, self.patchWidth = pHeight, pWidth

            SUM_KERNEL = np.ones((pHeight, pWidth), dtype=np.uint8)
            convolvedMat = cv2.filter2D(
                self.originalSourceRegion, cv2.CV_8U, SUM_KERNEL, anchor=(0, 0))

            # Identify potential upper-left corners of source patches
            area = pHeight * pWidth
            self.sourcePatchULList = np.argwhere(
                convolvedMat[:-pHeight, :-pWidth] == area)

        # Filter potential patch origins to be within a distance threshold of (aX, aY)
        max_distance = 120
        UL_array = np.array(self.sourcePatchULList)
        distances = np.sqrt((UL_array[:, 1] - aX)
                            ** 2 + (UL_array[:, 0] - aY) ** 2)
        self.sourcePatchULList = UL_array[distances <= max_distance]

        # Determine the portion of the target patch that corresponds to the source region.
        targetPatch = self.sourceRegion[aY:aY + pHeight, aX:aX + pWidth]
        source_pixels = targetPatch == 1
        self.targetPatchSList = np.argwhere(source_pixels)
        self.targetPatchTList = np.argwhere(~source_pixels)

        # Prepare to calculate the error and variance of each candidate patch.
        countedNum = float(len(self.targetPatchSList))
        minError = bestPatchVariance = sys.maxsize
        alpha, beta = 0.9, 0.5

        SList_i_indices = self.targetPatchSList[:, 0]
        SList_j_indices = self.targetPatchSList[:, 1]
        image_float32 = np.float32(self.image)
        image_int = self.image.astype(np.int64)
        targetPixels = image_int[SList_i_indices + aY, SList_j_indices + aX, :]

        # Evaluate each potential source patch for suitability.
        for (y, x) in self.sourcePatchULList:
            sourcePixels = image_int[SList_i_indices +
                                     y, SList_j_indices + x, :]
            differences = sourcePixels - targetPixels
            patchError = np.sum(differences ** 2) / countedNum

            # Evaluate error and variance for the current candidate.
            if alpha * patchError <= minError:
                meanRGB = np.mean(sourcePixels, axis=0)

                i_indices = self.targetPatchTList[:, 0]
                j_indices = self.targetPatchTList[:, 1]

                sourcePixels = image_float32[i_indices + y, j_indices + x, :]
                sourcePixels = sourcePixels - meanRGB

                patchVariance = np.sum(sourcePixels ** 2)

                # Use alpha & Beta to encourage path with less patch variance.
                # For situations in which you need little variance.
                # Alpha = Beta = 1 to disable.
                # Update best match if the current patch is better.
                if patchError < alpha * minError or patchVariance < beta * bestPatchVariance:
                    bestPatchVariance = patchVariance
                    minError = patchError
                    self.bestMatchUpperLeft = (x, y)
                    self.bestMatchLowerRight = (x+pWidth-1, y+pHeight-1)

    def updateMats(self):
        """
        Updates matrices by copying patch data from the source region to the 
        target region and updating gradients, confidence, and region masks.
        """
        # Get the coordinates of the target point and the best matching upper left corner of the source patch.
        tX, tY = self.targetPoint
        (aX, aY), _ = self.getPatch(self.targetPoint)
        bulX, bulY = self.bestMatchUpperLeft

        # Convert list of tuples to a NumPy array and get i and j arrays
        indices = np.array(self.targetPatchTList)
        i_array, j_array = indices[:, 0], indices[:, 1]

        # Calculate the absolute positions of the source and target indices.
        source_indices = (bulY + i_array, bulX + j_array)
        target_indices = (aY + i_array, aX + j_array)

        # Copy the patch from the source to the target in the image, gradients, and confidence matrices.
        self.image[target_indices] = self.image[source_indices]
        self.gradientX[target_indices] = self.gradientX[source_indices]
        self.gradientY[target_indices] = self.gradientY[source_indices]
        self.confidence[target_indices] = self.confidence[tY, tX]

        # Update the source and target region masks and reset the mask at the target indices.
        self.sourceRegion[target_indices] = 1
        self.targetRegion[target_indices] = 0
        self.mask[target_indices] = 0

    def checkEnd(self):
        """
        Checks if the inpainting process should end, which is true when there are no more target regions to fill.
        """
        # Return True if all regions are now considered source regions (i.e., inpainting complete).
        return np.all(self.sourceRegion != 0)


def inpaint(image, mask, patch_width=Inpainter.DEFAULT_HALF_PATCH_WIDTH):
    i = Inpainter(image, mask, patch_width)
    i.inpaint()
    return i.result
