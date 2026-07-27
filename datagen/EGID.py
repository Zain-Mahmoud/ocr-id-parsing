#dependencies you might not have:
# install: pip install --upgrade arabic-reshaper
# install: pip install python-bidi
#ref:https://gist.github.com/amrza/04658c71ac02d82580855f89b9b3dff4

#imports
import sys
import yaml
import glob
from random import randint,choice
from PIL import Image,ImageDraw,ImageFont
import arabic_reshaper
from bidi.algorithm import get_display
import os
import numpy as np
import csv
from pathlib import Path
import albumentations as A
import cv2

def convert_digits(text):
    western = "0123456789"
    eastern = "٠١٢٣٤٥٦٧٨٩"

    table = str.maketrans(western, eastern)

        
    return text.translate(table)

class GenerateID():

    def __init__(self,images_num,augment=False,augment_batches=10, type_="train") :
        self.images_num=images_num
        self.augment=augment
        self.augment_batches=augment_batches
        self.type_ = type_
        self.loadFiles()

        for i in range(images_num):
            self.generateID(f"../data/synthetic-ids/{self.type_}/images/ID{i}.png",f"../data/synthetic-ids/{self.type_}/images/IDB{i}.png", f"../data/synthetic-ids/{self.type_}/IDLabels.csv", id=i)
            print('█' * int(i/(images_num-1)*20) + '░' * int(20-(i/(images_num-1) *20)), end="\r")
        print(f"\ngenerated {images_num} ID")
        if self.augment:
            self.dataAugmentation()

    def generateID(self,front_output_path,back_output_path, label_path, id):
        if self.augment:
            front_output_path = f"../data/synthetic-ids/{self.type_}/images/augment/ID{id}.png"
            back_output_path = f"../data/synthetic-ids/{self.type_}/images/augment/IDB{id}.png"
            label_path = f'../data/synthetic-ids/{self.type_}/augment_IDLabels.csv'

        person=self.generateName()
        first_name=person["first_name"]
        paternal_names=person['paternal_names']
        gender=person['gender']
        religion = person['religion']
        marital_status = person['marital_status']
        birthdate=self.generateBirthdate()
        birthdate_formatted="{YY}/{MM}/{DD}".format(YY=birthdate[-4:],MM=birthdate[2:4],DD=birthdate[:2])
        id_number=self.generateIDNumber(birthdate)
        governorate, station_line = self.generateGovStation()
        address = self.generateAddress()

        first_name_image =self.putArText('./IDBLANK.jpg',25,first_name,(850,420))
        self.putArText(first_name_image,25,paternal_names,(850,460))
        self.putArText(first_name_image,25,address,(850,500))
        self.putArText(first_name_image,25,station_line,(850,550))
        self.putArNumText(first_name_image,25,birthdate_formatted,(400,630))
        self.putArNumText(first_name_image,25,id_number,(850,650))

        first_name_image.paste(self.generateImage(gender), (240,365))
        first_name_image.save(front_output_path)

        #back id
        issue_date=self.generateIssueDate()
        expiration_date = self.generateExpirationDate(issue_date)

        back_id=self.putArNumText('./IDBACKBLANK.jpg',25,id_number,(750,345))
        self.putArNumText(back_id,25,issue_date,(480,345))
        exp_text="البطاقة سارية حتى"
        job_text = self.generateJobTitle(gender)
        self.putArText(back_id,22,exp_text,(760,510))
        self.putArNumText(back_id,25,expiration_date,(570,520))
        self.putArText(back_id,22,job_text,(750,370))
        if gender==1:
            gender_text="ذكر"
        else:
            gender_text="أنثى"
        self.putArText(back_id,22,gender_text,(750,435))   
        self.putArText(back_id, 22, religion, (650, 435))
        self.putArText(back_id, 22, marital_status, (500, 435))

        back_id.save(back_output_path)
        f = open(label_path, "a", encoding="utf-8", newline='')

        national_id_clean = convert_digits(id_number.replace(' ', ''))

        front_labels = {
            "image": f"ID{id}.png",
            "side": "front",
            "first_name": first_name,
            "last_name": paternal_names,
            "gender": None,
            "national_id": national_id_clean,   # national_id appears on both sides
            "issue_date": None,
            "expiration_date": None,
            "address": address,
            "address2": station_line,
            "job_title": None,
            "religion": None,
            "marital_status": None
        }

        # Fields present on the BACK of the card only get real values here;
        # front-only fields are explicitly None.
        back_labels = {
            "image": f"IDB{id}.png",
            "side": "back",
            "first_name": None,
            "last_name": None,
            "gender": gender_text,
            "national_id": national_id_clean,   # national_id appears on both sides
            "issue_date": convert_digits(issue_date),
            "expiration_date": convert_digits(expiration_date),
            "address": None,
            "address2": None,
            "job_title": job_text,
            "religion": religion,
            "marital_status": marital_status
        }

        rows_to_write = [front_labels, back_labels]

        if self.augment:
            rows_to_write = []
            for c in range(self.augment_batches):
                aug_front = dict(front_labels)
                aug_front["image"] = f"a_ID{id}_{c+1}.png"
                rows_to_write.append(aug_front)

                aug_back = dict(back_labels)
                aug_back["image"] = f"a_IDB{id}_{c+1}.png"
                rows_to_write.append(aug_back)

        writer = csv.DictWriter(f, fieldnames=list(front_labels.keys()))
        if os.stat(label_path).st_size == 0 and not self.augment:
            writer.writeheader()
        writer.writerows(rows_to_write)
        f.close()


    def generateJobTitle(self, gender):
        if gender == 1:
            return choice(self.male_jobs)
        return choice(self.female_jobs)

    def loadFiles(self):
        #loading files: names, governorates, police stations, addresses, and image paths
        with open('names.yaml', 'r', encoding='utf-8') as file:
            names_dict = yaml.safe_load(file)

        with open('jobs.yaml', 'r', encoding='utf-8') as f2:
            jobs_dict = yaml.safe_load(f2)

        self.male_mus_names=names_dict["male_names"]["muslim"]
        self.male_chr_names=names_dict["male_names"]["christian"]

        self.female_mus_names=names_dict["female_names"]["muslim"]
        self.female_chr_names=names_dict["female_names"]["christian"]

        with open('egy_gov.yaml', 'r', encoding='utf-8') as file:
            gov_dict = yaml.safe_load(file)
        self.gov_names = gov_dict['egyptian_governorates']

        # NEW: police stations per governorate
        with open('police_stations.yaml', 'r', encoding='utf-8') as file:
            station_dict = yaml.safe_load(file)
        self.police_stations = station_dict.get('police_stations', {})

        # NEW: address components
        with open('addresses.yaml', 'r', encoding='utf-8') as file:
            address_dict = yaml.safe_load(file)
        self.street_names = address_dict['street_names']
        self.street_prefixes = address_dict['street_prefixes']
        self.area_descriptors = address_dict['area_descriptors']

        self.female_images=glob.glob("./images/female/**")
        self.male_images=glob.glob("./images/male/**")

        self.male_jobs = jobs_dict['male_jobs']
        self.female_jobs = jobs_dict['female_jobs']


    def generateBirthdate(self):
        '''
        Generates random Birthdate
        Args:
            None
        Returns:
            birthdate(str) in format DDMMYYYY
        '''
        day=randint(1,31)
        month=randint(1,12)
        year=randint(1961,2008)
        
        birthdate="{:02}{:02}{}".format(day,month,year)
        return birthdate

    def generateIDNumber(self,birthdate):
        '''
        Generates random ID number based on a birthdate
        Args:
            birthdate(str): DDMMYYYY
        Returns:
            id_number(str): {century_indicator} {YY} {MM} {DD} {tailing_number}
        '''
        birthdate_year=birthdate[-4:]
        if int(birthdate_year)<2000:
            century_indicator="2"
        else:
            century_indicator="3"
        YY=birthdate[-2:]
        MM=birthdate[2:4]
        DD=birthdate[:2]
        tailing_number=str(randint(1000000,9999999))
        id_number=f"{century_indicator} {YY} {MM} {DD} {tailing_number}"
        return id_number 

    def generateGov(self):
        '''
        chooses a random governorate from gov_names list
        Args:
            None
        Returns:
            governorate name(str)
        '''
        return choice(self.gov_names)

    def generateGovStation(self):
        '''
        Chooses a random governorate and a plausible police station within it,
        formatted the way Egyptian national IDs display it:
        "{station} - {governorate}"

        Falls back to a generic ordinal station name ("قسم أول {governorate}")
        if the governorate has no explicit station list in police_stations.yaml.

        Args:
            None
        Returns:
            tuple: (governorate(str), station_line(str))
        '''
        governorate = choice(self.gov_names)
        stations = self.police_stations.get(governorate)

        if not stations:
            # Fallback for governorates not yet populated in police_stations.yaml
            stations = [
                f"قسم أول {governorate}",
                f"قسم ثاني {governorate}",
                f"مركز {governorate}",
            ]

        station = choice(stations)
        station_line = f"{station} - {governorate}"
        return governorate, station_line

    def generateName(self):
        '''
        generates male and female names randomly drawn from male_names and female_names lists
        Args:
            None
        Returns:
            dict with keys: first_name, paternal_names, gender
        '''
        #baby is a girl or a boy?
        gender=randint(0,1)
        religion = randint(0,1)

        if gender==1:
            if religion == 1:
                first_name = choice(self.male_mus_names)
                religion_text = "مسلم"
            else:
                first_name = choice(self.male_chr_names)
                religion_text = "مسيحي"
        else:
            if religion == 1:
                first_name = choice(self.female_mus_names)
                religion_text = "مسلمة"
            else:
                first_name = choice(self.female_chr_names)
                religion_text = "مسيحية"

        married = randint(0, 2)
        if married == 0:
            marital_status = 'متزوج' if gender == 1 else 'متزوجة'
        if married == 1:
            marital_status = 'اعزب' if gender == 1 else 'أنسة'
        if married == 2:
            marital_status = 'أرمل' if gender == 1 else 'أرملة'

        paternal_names=""
        for i in range(4):
            par = self.male_mus_names if religion == 1 else self.male_chr_names
            paternal_names+= choice(par)+" "
        
        return {'first_name':first_name, 'paternal_names':paternal_names,'gender':gender, 'religion': religion_text, 'marital_status': marital_status}

    def generateImage(self,gender):
        '''
        chooses a random image based on gender
        Args:
            gender(int): male=1, female=0
        Returns:
            pil grayscale image with size (175,175)
        '''

        if gender==1:
            return Image.open(choice(self.male_images)).resize((175,175)).convert("L")
        else:
            return Image.open(choice(self.female_images)).resize((175,175)).convert("L")

    def generateAddress(self):
        '''
        Generates a randomized Egyptian-style street address, e.g.:
        "12 شارع النصر" or "5 ش فيصل متفرع من الهرم"

        Args:
            None
        Returns:
            address(str)
        '''
        building_number = convert_digits(str(randint(1, 250)))
        prefix = choice(self.street_prefixes)
        street = choice(self.street_names)
        descriptor = choice(self.area_descriptors)

        if descriptor:
            second_street = choice([s for s in self.street_names if s != street])
            address = f"{building_number} {prefix} {street} {descriptor} {second_street}"
        else:
            address = f"{building_number} {prefix} {street}"

        return address

    def putArText(self,img,font_size,text,position):
        '''
        writes Arabic text on an image
        Args:
            img: either an image path(str) or pil image
            font_size(int)
            text(str)
            position(tuple or list)
        Returns
            pil image
        '''
        if type(img)==str:
            self.image = Image.open(img)
        else:
            self.image=img
        font = ImageFont.truetype('/Users/zain/fonts/Noto_Sans_Arabic/static/NotoSansArabic-Medium.ttf', font_size)
        reshaped_text = arabic_reshaper.reshape(text)  # correct its shape
        bidi_text = get_display(reshaped_text)  # correct its direction
        # start drawing on image
        draw = ImageDraw.Draw(self.image)
        text_length = draw.textlength(bidi_text, font=font)
        draw.text((position[0]-text_length, position[1]), bidi_text, (0, 0, 0), font=font)
        
        return self.image
        


    def putArNumText(self,img,font_size,text,position):
        '''
        writes Eastern Arabic Numerals on an image
        Args:
            img: either an image path(str) or pil image
            font_size(int)
            text(str)
            position(tuple or list)
        Returns
            image
        '''
        if type(img)==str:
            self.image = Image.open(img)
        else:
            self.image=img
        font = ImageFont.truetype("/Users/zain/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)

        western_to_eastern_ar_numerals = {
            '0': '٠',
            '1': '١',
            '2': '٢',
            '3': '٣',
            '4': '٤',
            '5': '٥',
            '6': '٦',
            '7': '٧',
            '8': '٨',
            '9': '٩',
            " ": " "
        }
        eastern_arabic = ''.join([western_to_eastern_ar_numerals.get(char, char) for char in text])
        
        # start drawing on image
        draw = ImageDraw.Draw(self.image)
        text_length = draw.textlength(eastern_arabic, font=font)
        draw.text((position[0]-text_length, position[1]), eastern_arabic, (0, 0, 0), font=font)
        
        return self.image

    def generateIssueDate(self):
        year=randint(2016,2023)
        month=randint(1,12)
        return "{:02}/{:02}".format(year,month)

    def generateExpirationDate(self,issue_date):
        return f"{int(issue_date[:4])+7}/{int(issue_date[5:7])-1:02}/31"


    def dataAugmentation(self):

        transform = A.Compose([
            A.SafeRotate(
                limit=(-90, 90),
                border_mode=cv2.BORDER_REPLICATE,
                p=1.0,
            ),
            A.Affine(
                translate_percent=(-0.2, 0.2),
                scale=(0.9, 1.1),
                shear=(-10, 10),
                border_mode=cv2.BORDER_REPLICATE,
                p=1.0,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.0,
                p=1.0,
            ),
        ])

        ids_dir = Path(f"../data/synthetic-ids/{self.type_}/images/augment")
        out_dir = Path(f"../data/synthetic-ids/{self.type_}/images")


        image_extensions = {".png", ".jpg", ".jpeg"}

        images = [
            p for p in ids_dir.iterdir()
            if p.suffix.lower() in image_extensions
            and not p.stem.startswith("a_")      # Don't augment previous augmentations
        ]

        for img_path in images:

            image = cv2.imread(str(img_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            stem = img_path.stem
            suffix = img_path.suffix

            for i in range(self.augment_batches):

                augmented = transform(image=image)["image"]

                augmented = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)

                cv2.imwrite(
                    str(out_dir / f"a_{stem}_{i+1}{suffix}"),
                    augmented,
                )

        print(f"Generated {self.augment_batches} augmentations per image.")




if __name__ == '__main__':
    type_ = sys.argv[1]

    size = int(sys.argv[2])
    augment = sys.argv[3]
    augment = augment == "yes"

    if augment:
        augment_batch_size = int(sys.argv[4])
        GenerateID(size, augment_batches=augment_batch_size, augment=augment, type_=type_)
    else:
        GenerateID(size, type_=type_)