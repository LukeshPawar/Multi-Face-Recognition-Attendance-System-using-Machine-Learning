import face_recognition as frg
import pickle as pkl 
import os 
import cv2 
import numpy as np
import yaml
from collections import defaultdict
from functools import lru_cache

information = defaultdict(dict)
cfg = yaml.load(open('config.yaml','r'),Loader=yaml.FullLoader)
DATASET_DIR = cfg['PATH']['DATASET_DIR']
PKL_PATH = cfg['PATH']['PKL_PATH']

@lru_cache(maxsize=1)
def get_databse():
    if not os.path.exists(PKL_PATH):
        with open(PKL_PATH,'wb') as f:
            pkl.dump({}, f)
    with open(PKL_PATH,'rb') as f:
        database = pkl.load(f)
    return database

@lru_cache(maxsize=1)
def _get_known_encodings_and_keys():
    database = get_databse()
    keys = list(database.keys())
    encodings = [database[key]['encoding'] for key in keys]
    return encodings, keys

def _clear_caches():
    try:
        get_databse.cache_clear()
    except Exception:
        pass
    try:
        _get_known_encodings_and_keys.cache_clear()
    except Exception:
        pass

def recognize(image, TOLERANCE, is_bgr=None): 
    database = get_databse()
    known_encoding, key_order = _get_known_encodings_and_keys()
    name = 'Unknown'
    id = 'Unknown'
    # Convert to RGB when needed
    if is_bgr is True:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif is_bgr is False:
        rgb_image = image
    else:
        # Fallback: assume input from OpenCV
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_locations = frg.face_locations(rgb_image, number_of_times_to_upsample=0)
    face_encodings = frg.face_encodings(rgb_image, face_locations, num_jitters=1)
    for (top,right,bottom,left),face_encoding in zip(face_locations,face_encodings):
        matches = frg.compare_faces(known_encoding,face_encoding,tolerance=TOLERANCE)
        distance = frg.face_distance(known_encoding,face_encoding)
        name = 'Unknown'
        id = 'Unknown'
        if True in matches:
            match_index = matches.index(True)
            matched_key = key_order[match_index]
            name = database[matched_key]['name']
            id = database[matched_key]['id']
            distance = round(float(distance[match_index]),2)
            cv2.putText(rgb_image,str(distance),(left,top-30),cv2.FONT_HERSHEY_SIMPLEX,0.75,(0,255,0),2)
        cv2.rectangle(rgb_image,(left,top),(right,bottom),(0,255,0),2)
        cv2.putText(rgb_image,name,(left,top-10),cv2.FONT_HERSHEY_SIMPLEX,0.75,(0,255,0),2)
    return rgb_image, name, id

def isFaceExists(image): 
    # Ensure RGB for face_recognition
    if image is None:
        return False
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_location = frg.face_locations(rgb_image, number_of_times_to_upsample=0)
    if len(face_location) == 0:
        return False
    return True

def submitNew(name, id, image, old_idx=None):
    database = get_databse()
    #Read image 
    if type(image) != np.ndarray:
        image = cv2.imdecode(np.frombuffer(image.read(), np.uint8), 1)

    isFaceInPic = isFaceExists(image)
    if not isFaceInPic:
        return -1
    #Encode image
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    encoding = frg.face_encodings(rgb_image, num_jitters=1)[0]
    #Append to database
    #check if id already exists
    existing_id = [database[i]['id'] for i in database.keys()]
    #Update mode 
    if old_idx is not None: 
        new_idx = old_idx
    #Add mode
    else: 
        if id in existing_id:
            return 0
        if len(database.keys()) > 0:
            try:
                new_idx = max(int(k) for k in database.keys()) + 1
            except Exception:
                new_idx = len(database)
        else:
            new_idx = 0
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    database[new_idx] = {'image':image,
                        'id': id, 
                        'name':name,
                        'encoding':encoding}
    with open(PKL_PATH,'wb') as f:
        pkl.dump(database,f)
    _clear_caches()
    return True

def get_info_from_id(id): 
    database = get_databse() 
    for idx, person in database.items(): 
        if person['id'] == id: 
            name = person['name']
            image = person['image']
            return name, image, idx       
    return None, None, None

def deleteOne(id):
    database = get_databse()
    id = str(id)
    key_to_delete = None
    for key, person in database.items():
        if str(person['id']) == id:
            key_to_delete = key
            break
    if key_to_delete is not None:
        del database[key_to_delete]
    with open(PKL_PATH,'wb') as f:
        pkl.dump(database,f)
    _clear_caches()
    return True

def build_dataset():
    counter = 0
    for image in os.listdir(DATASET_DIR):
        image_path = os.path.join(DATASET_DIR,image)
        image_name = image.split('.')[0]
        parsed_name = image_name.split('_')
        person_id = parsed_name[0]
        person_name = ' '.join(parsed_name[1:])
        if not image_path.endswith('.jpg'):
            continue
        image = frg.load_image_file(image_path)
        information[counter]['image'] = image 
        information[counter]['id'] = person_id
        information[counter]['name'] = person_name
        information[counter]['encoding'] = frg.face_encodings(image, num_jitters=1)[0]
        counter += 1

    with open(os.path.join(DATASET_DIR,'database.pkl'),'wb') as f:
        pkl.dump(information,f)
    _clear_caches()

if __name__ == "__main__": 
    deleteOne(4)

