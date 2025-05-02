#import logging
#logging.getLogger('boxmot').setLevel(logging.DEBUG)

import cv2
import numpy as np
from boxmot import DeepOcSort
from pathlib import Path
from ultralytics import YOLO
import psycopg2
import pandas as pd
from dash import Dash, html, dcc, callback, Input, Output
import plotly_express as px
import webbrowser
table_name='detections'

#connect to the database
db_params = {
'dbname':"trisense",
    'user':"ncerni",
    'host':"localhost",
    'port':"5432"
}

try:
    conn = psycopg2.connect(**db_params)
    cursor = conn.cursor()
    
    #check if the table exists
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id SERIAL PRIMARY KEY,
            frame_number INTEGER,
            x INTEGER,
            y INTEGER,
            width INTEGER,
            height INTEGER,
            probability FLOAT,
            class INTEGER,
            track_id INTEGER
        )
    """)
   # conn.commit()
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS histogram (
            id SERIAL PRIMARY KEY,
            frame_number INTEGER,
            bin_start INTEGER,
            bin_end INTEGER,
            count INTEGER
        )
    """)
   # conn.commit()
    
    cursor.execute(f"""DELETE FROM histogram; DELETE FROM {table_name}""")
    conn.commit()
    
except psycopg2.Error as e:
    print(f"Error connecting to the database: {e}")
        
def histo(frame,frameNumber):
    bwFrame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    hist=cv2.calcHist([bwFrame],[0],None,[256],[0,256])
    for bin in range(8,256,8):
        binSum=int(0)
        for subBin in range(bin-8,bin,1):
            binSum=binSum+int(hist[subBin])
        cursor.execute(f"INSERT INTO histogram (frame_number,bin_start,bin_end,count) VALUES ({frameNumber},{bin-8},{bin},{binSum})")
    conn.commit()
    
# Initialize BoxMOT tracker
model_weights = Path('osnet_x1_0_msmt17.pt')
tracker = DeepOcSort(
    reid_weights=model_weights,
    device='mps',  # mps for Mac, 'cuda' for GPU, 'cpu' for CPU
    half=False
)
 
# Initialize YOLO detector
yolo_model= YOLO('yolov8n.pt')

# Initialize video capture
cap = cv2.VideoCapture('cars.mp4')  # Try index 0; use 1 or a video file if needed
if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()


frame_num=0
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to read frame.")
        break

    histo(frame=frame,frameNumber=frame_num)
    # Get detections from YOLO
    results = yolo_model.predict(frame)
    detections = []
    if results[0].boxes is not None:
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
            scores = result.boxes.conf.cpu().numpy()  # Confidence scores
            classes = result.boxes.cls.cpu().numpy()  # Class IDs            
            for box, score, cls in zip(boxes, scores, classes):
                detections.append([box[0], box[1], box[2], box[3], score, cls])

    print('frame number:',frame_num)
    frame_num=frame_num+1
    detections = np.array(detections, dtype=np.float32) if detections else np.empty((0, 6), dtype=np.float32)

    # Update tracker
    tracks = tracker.update(detections, frame)

    # Draw tracks on frame
    for track in tracks:
        x1, y1, x2, y2, track_id, score = track[:6]
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        
        cv2.circle(frame,(int(x1+(x2-x1)/2),int(y1+(y2-y1)/2)),5,(255,0,0),2)
        
        cv2.putText(frame, f"ID: {int(track_id)}, score: {int(score*100)}%", (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cursor.execute(
                f"INSERT INTO {table_name} (frame_number, x, y, width, height, probability, class, track_id) "
                "VALUES (%s, %s, %s, %s, %s, %s,%s,%s)",
                (frame_num, int(x1), int(y1), int(x2-x1), int(y2-y1), float(score), int(cls),int(track_id)),
                    )   
           
        # Display frame
    conn.commit()
    cv2.imshow("Webcam", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
#cleanup

cap.release()
cv2.destroyAllWindows()
#some bug in openCV and unix systems, doestn't want to close openCV window on its own.
for i in range (1,5):
    cv2.waitKey(1)



#dash area

def getMaxTrackID():
    
    cursor.execute("SELECT MAX(track_id) FROM detections")
    return max(cursor.fetchone())
def  getMaxFrameNo():
    cursor.execute("SELECT MAX(frame_number) FROM histogram")
    return max(cursor.fetchone())
    
def fetch_data(track:int=1):
    df = pd.read_sql_query(f"SELECT frame_number, id, class, probability FROM detections WHERE track_id={track}", conn)
    return df

def fetch_histo(frame:int=1):
    
    df=pd.read_sql_query(f"SELECT bin_start, count FROM histogram WHERE frame_number={frame}", conn)
    return df

def run_dash():
    app=Dash()
    maxTrackId=getMaxTrackID()
    maxFrameNo=getMaxFrameNo()
    app.layout = html.Div([
        html.Div(children=[
            html.H1("Probability of detection for trackIDs"),
            dcc.Graph(id='probability-chart'),
            html.Label('Track ID:'),
            dcc.Dropdown(id='dropdown',
                    options=[{'label': str(i), 'value':i}for i in range(maxTrackId)])
            ],style={'padding': 10, 'flex': 1}),
        
        html.Div(children=[
        html.H1("BW histogram for a frame"),
        dcc.Graph(id="histogramChart"),
        html.Label('Frame No.'),
        dcc.Dropdown(id='frameDropdown',
                     options=[{'label': str(i), 'value':i}for i in range(maxFrameNo)])
    ],style={'padding':10, 'flex':1})],style={'display':'flex','flex_direction':'row'})
    
    webbrowser.open('localhost:8050')
    app.run(debug=False)

@callback(
    Output('probability-chart', 'figure'),
    Input('dropdown', 'value'))
def update_chart(input_value):
    #print(input_value)
    df = fetch_data(input_value)
    fig = px.line(df, x='frame_number', y='probability', 
            title='Probability of Detections Over Frames')
    return fig

@callback(
    Output('histogramChart','figure'),
    Input('frameDropdown','value'))
def update_histo(input_value):
    df=fetch_histo(input_value)
    print(df)
    fig=px.line(df,x='bin_start',y='count',title='histogram')
    return fig



if __name__=='__main__':
    
    run_dash()

