from decimal import Decimal
import cv2
import numpy as np
import glob


class TrayCoinAnalyzer:
    def __init__(self, image_dir):
        self.image_paths = glob.glob(f'{image_dir}/*.jpg')
        self.images = [cv2.imread(path) for path in self.image_paths]
        if any(img is None for img in self.images):
            raise ValueError("One or more images could not be loaded pr"
                             "operly.")
        self.results = []

    def preprocess_image(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        contrast_enhanced = clahe.apply(gray)
        blurred = cv2.GaussianBlur(contrast_enhanced, (3, 3), 0)
        edged = cv2.Canny(blurred, 81, 209)
        kernel = np.ones((4, 4), np.uint8)
        closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
        return closed

    def detect_coins(self, image, edged):
        circles = cv2.HoughCircles(edged, cv2.HOUGH_GRADIENT, dp=1.1, minDist=20,
                                   param1=20, param2=30, minRadius=15, maxRadius=40)
        coins = []
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                if self.validate_coin((x, y, r), image):
                    area = np.pi * (r ** 2)
                    coin_type = '5 PLN' if r > 32.9 else '5 gr'
                    on_tray = self.is_coin_on_tray((x, y), image)
                    coins.append({'center': (x, y), 'radius': r, 'area': Decimal(area).quantize(Decimal('0.01')),
                                  'type': coin_type, 'on_tray': on_tray})
        return coins

    def validate_coin(self, circle, image):
        x, y, r = circle
        coin_img = image[max(0, y - r):y + r, max(0, x - r):x + r]
        if coin_img.size == 0:
            return False
        gray_coin = cv2.cvtColor(coin_img, cv2.COLOR_BGR2GRAY)
        if cv2.mean(gray_coin)[0] < 50:
            return False
        return True

    def is_coin_on_tray(self, center, image, radius=5):
        # Automatic color range adjustment based on the image
        lower_orange, upper_orange = self.adjust_orange_hue(image)

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_orange, upper_orange)

        if (0 <= center[0] < image.shape[1]) and (0 <= center[1] < image.shape[0]):
            x_start = max(center[0] - radius, 0)
            x_end = min(center[0] + radius, image.shape[1] - 1)
            y_start = max(center[1] - radius, 0)
            y_end = min(center[1] + radius, image.shape[0] - 1)
            region = mask[y_start:y_end, x_start:x_end]
            return np.any(region > 0)
        else:
            return False

    def adjust_orange_hue(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hue_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
        peak = np.argmax(hue_hist)
        lower_orange = np.array([max(0, peak - 8), 80, 100])
        upper_orange = np.array([min(180, peak + 8), 255, 255])
        return lower_orange, upper_orange

    def find_trays(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_hue = np.array([12, 129, 129])
        upper_hue = np.array([25, 255, 255])
        mask = cv2.inRange(hsv, lower_hue, upper_hue)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            tray_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(tray_contour)
            return {'contour': tray_contour, 'area': Decimal(area).quantize(Decimal('0.01'))}
        return None

    def analyze_images(self):
        for img in self.images:
            edged = self.preprocess_image(img)
            tray = self.find_trays(img)
            coins = self.detect_coins(img, edged)
            coin_img = img.copy()
            total_value_on_tray = Decimal('0')
            total_value_off_tray = Decimal('0')
            count_5_pln = 0
            count_5_gr = 0
            area_5_pln = Decimal('0')
            area_5_gr = Decimal('0')

            for coin in coins:
                cv2.circle(coin_img, coin['center'], coin['radius'], (0, 255, 0), 2)
                cv2.putText(coin_img, coin['type'], (coin['center'][0] - 20, coin['center'][1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                if coin['on_tray']:
                    if coin['type'] == '5 PLN':
                        total_value_on_tray += Decimal('5')
                        count_5_pln += 1
                        area_5_pln += coin['area']
                    else:
                        total_value_on_tray += Decimal('0.05')
                        count_5_gr += 1
                        area_5_gr += coin['area']
                else:
                    if coin['type'] == '5 PLN':
                        total_value_off_tray += Decimal('5')
                        count_5_pln += 1
                        area_5_pln += coin['area']
                    else:
                        total_value_off_tray += Decimal('0.05')
                        count_5_gr += 1
                        area_5_gr += coin['area']

            if tray:
                ratio_5_pln_to_tray = (area_5_pln / tray['area']).quantize(Decimal('0.0001')) if tray[
                                                                                                     'area'] != 0 else Decimal(
                    '0')
            else:
                ratio_5_pln_to_tray = Decimal('0')

            self.results.append({
                'original': img,
                'processed': coin_img,
                'edges': edged,
                'coins': coins,
                'tray_area': tray['area'] if tray else Decimal('0'),
                'total_value_on_tray': total_value_on_tray,
                'total_value_off_tray': total_value_off_tray,
                'count_5_pln': count_5_pln,
                'count_5_gr': count_5_gr,
                'area_5_pln': area_5_pln,
                'area_5_gr': area_5_gr,
                'ratio_5_pln_to_tray': ratio_5_pln_to_tray
            })

    def display_results(self):
        for index, result in enumerate(self.results):
            print(f"Image {index + 1}:")
            print(f"Total value on Tray: {result['total_value_on_tray']} PLN")
            print(f"Total value off Tray: {result['total_value_off_tray']} PLN")
            print(f"Number of 5 PLN coins: {result['count_5_pln']}, Total area: {result['area_5_pln']}")
            print(f"Number of 5 gr coins: {result['count_5_gr']}, Total area: {result['area_5_gr']}")
            print(f"Tray area: {result['tray_area']}")
            print(f"5 PLN coins are {result['ratio_5_pln_to_tray']:.4f} times smaller than the tray area.")
            cv2.imshow(f"Processed Image {index + 1}", result['processed'])
            cv2.waitKey(0)  # Wait indefinitely until a key is pressed
        cv2.destroyAllWindows()


# Example usage
image_directory = '/Users/lena/PycharmProjects/WMA2/images'
analyzer = TrayCoinAnalyzer(image_directory)
analyzer.analyze_images()
analyzer.display_results()
