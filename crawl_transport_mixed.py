import os
from icrawler.builtin import GoogleImageCrawler, BingImageCrawler
from PIL import Image, ImageEnhance, ImageOps
import imagehash
from tqdm import tqdm
import time
import random

# 저장 경로
OUT_DIR = '/home/parkwooyeol/workspace/Resnet-Experiment/downloads'
os.makedirs(OUT_DIR, exist_ok=True)

# 클래스별 키워드
CLASSES = {
    'bus': ['bus', 'city bus', 'school bus', '버스', '시내버스', '고속버스'],
    'taxi': ['taxi', 'yellow taxi', '택시', '서울택시', '택시차량'],
    'motorcycle': ['motorcycle', 'sports motorcycle', '오토바이', '스포츠오토바이', '배달오토바이'],
    'bicycle': ['bicycle', 'mountain bike', '자전거', '산악자전거', '도로자전거'],
    'truck': ['truck', 'pickup truck', '트럭', '덤프트럭', '화물트럭'],
    'e_scooter': ['electric scooter', '전동킥보드', 'escooter', '전기스쿠터'],
    'police_car': ['police car', '경찰차', 'cop car', '순찰차'],
    'ambulance': ['ambulance', '구급차', 'emergency ambulance', '응급차량'],
    'van': ['van', '승합차', 'minivan', '미니밴'],
    'construction_vehicle': ['construction vehicle', '건설차량', 'bulldozer', '크레인', '덤프카']
}

NUM_PER_CLASS = 1000
MIN_WIDTH, MIN_HEIGHT = 128, 128  # 학습용 최소 크기
CRAWLERS = [GoogleImageCrawler, BingImageCrawler]

# ---------- 이미지 크롤링 ----------
def crawl_class(label, keywords, num_images=NUM_PER_CLASS):
    target_dir = os.path.join(OUT_DIR, label)
    os.makedirs(target_dir, exist_ok=True)

    per_keyword = max(1, num_images // len(keywords))
    for kw in keywords:
        for Crawler in CRAWLERS:
            print(f'[+] Crawling "{kw}" for {label} using {Crawler.__name__}...')
            crawler = Crawler(storage={'root_dir': target_dir})
            crawler.crawl(keyword=kw, max_num=per_keyword, min_size=(MIN_WIDTH, MIN_HEIGHT))

# ---------- 중복 제거 & 리사이즈 ----------
def remove_duplicates_and_resize(target_size=(128,128)):
    print('[*] Removing duplicates and resizing...')
    for label in CLASSES.keys():
        folder = os.path.join(OUT_DIR, label)
        if not os.path.isdir(folder):
            continue
        hashes = {}
        removed = 0
        for fname in tqdm(os.listdir(folder), desc=label):
            path = os.path.join(folder, fname)
            try:
                img = Image.open(path).convert('RGB')
                if img.width < MIN_WIDTH or img.height < MIN_HEIGHT:
                    os.remove(path); removed += 1; continue
                h = str(imagehash.phash(img))
                if h in hashes:
                    os.remove(path); removed += 1
                else:
                    hashes[h] = path
                    img = img.resize(target_size)
                    img.save(path)
            except:
                try: os.remove(path)
                except: pass
                removed += 1
        print(f'  removed {removed} images from {label}')

# ---------- 데이터 증강 ----------
def augment_image(img):
    ops = [
        lambda x: x.transpose(Image.FLIP_LEFT_RIGHT),  # 좌우반전
        lambda x: x.transpose(Image.FLIP_TOP_BOTTOM),  # 상하반전
        lambda x: ImageEnhance.Brightness(x).enhance(random.uniform(0.7, 1.3)),  # 밝기
        lambda x: ImageEnhance.Contrast(x).enhance(random.uniform(0.7, 1.3)),    # 대비
        lambda x: ImageOps.solarize(x, threshold=random.randint(100,200))        # 솔라라이즈
    ]
    op = random.choice(ops)
    return op(img)

def augment_class(label, target_num=NUM_PER_CLASS):
    folder = os.path.join(OUT_DIR, label)
    if not os.path.isdir(folder):
        return
    images = os.listdir(folder)
    count = len(images)
    idx = 0
    while count < target_num and idx < len(images)*5:  # 5배까지 증강 허용
        img_path = os.path.join(folder, images[idx % len(images)])
        try:
            img = Image.open(img_path).convert('RGB')
            aug = augment_image(img)
            new_path = os.path.join(folder, f'aug_{count}.jpg')
            aug.save(new_path)
            count += 1
        except:
            pass
        idx += 1
    print(f'[*] Augmented {label} to {count} images')

# ---------- 부족하면 자동 재시도 ----------
def retry_missing_images(target_num=NUM_PER_CLASS):
    for label in CLASSES.keys():
        folder = os.path.join(OUT_DIR, label)
        count = len(os.listdir(folder)) if os.path.exists(folder) else 0
        attempt = 0
        while count < target_num and attempt < 3:  # 최대 3회 재시도
            missing = target_num - count
            print(f'[*] {label} has {count} images, retrying to get {missing} more...')
            crawl_class(label, CLASSES[label], num_images=missing)
            remove_duplicates_and_resize()
            count = len(os.listdir(folder))
            attempt += 1
            time.sleep(1)
        # 재시도 후에도 부족하면 증강으로 채우기
        if count < target_num:
            augment_class(label, target_num)

if __name__ == '__main__':
    # 1차 크롤링
    for label, keywords in CLASSES.items():
        crawl_class(label, keywords, NUM_PER_CLASS)

    # 중복 제거 & 리사이즈
    remove_duplicates_and_resize()

    # 부족하면 재시도 + 증강
    retry_missing_images()