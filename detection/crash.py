import cv2
import os
import numpy as np
from .tracker import CentroidTracker
from math import acos, pi
from collections import OrderedDict
from scipy.spatial import distance as dist
import torch

class CarCrashDetector:
    def __init__(self, model, fps=10.0, frame_inc=5, risk_thresh=0.6, accident_thresh=0.8):
        self.model = model
        self.fps = fps
        self.frame_inc = frame_inc
        self.T = frame_inc/fps
        self.ct = CentroidTracker()
        self.centroids = {}
        self.accelerations = {}
        self.norm_diffs = {}
        self.prev_speeds = {}
        self.overlaps = {}
        self.angles = {}
        self.risk_thresh = risk_thresh
        self.accident_thresh = accident_thresh
        self.frame_scores = {}
        self.accident_moments = []
        self.all_frames = {}
        self.velocities = {}
        self.collision_details = {}
        self.pixel_to_meter_ratio = 20.0
        self.class_names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}
        self.vehicle_classes = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck', 1: 'bicycle' , 0: 'person',16: 'dog',19: 'cow'}
    def process_frame(self, frame, idx):
        results = self.model(frame[..., ::-1])[0]
        boxes = []
        class_infos = []
        for *xyxy, conf, cls in results.boxes.data.cpu().numpy():
            x1,y1,x2,y2 = map(int,xyxy)
            cls_id = int(cls)
            if conf > 0.5 and cls_id in self.vehicle_classes:
                boxes.append([x1,y1,x2,y2])
                class_infos.append({'class_id': cls_id,'class_name': self.vehicle_classes[cls_id],'confidence': conf})
        objs = self.ct.update(boxes, class_infos)
        h,w = frame.shape[:2]
        for oid,cent in objs.items():
            self.centroids.setdefault(oid, []).append(cent)
        self._motion(idx,w,h)
        self._overlaps_angles(idx)
        return frame, objs, boxes
    def _motion(self, idx, w, h):
        intv,th= self.frame_inc,15.0
        for oid, cents in self.centroids.items():
            if len(cents)>intv:
                dx = cents[-1][0]-cents[-1-intv][0]
                dy = cents[-1][1]-cents[-1-intv][1]
                mag = np.hypot(dx,dy)
                if mag==0: continue
                norm=(dx/mag,dy/mag)
                if mag>th:
                    self.norm_diffs.setdefault(oid,[]).append([norm,mag,idx])
                vx_pix_per_frame = dx/(self.T*intv)
                vy_pix_per_frame = dy/(self.T*intv)
                vx_ms = vx_pix_per_frame / self.pixel_to_meter_ratio
                vy_ms = vy_pix_per_frame / self.pixel_to_meter_ratio
                speed_ms = np.hypot(vx_ms, vy_ms)
                self.velocities.setdefault(oid, []).append({'frame': idx,'vx': vx_ms,'vy': vy_ms,'speed': speed_ms,'direction': norm})
                swx = vx_pix_per_frame*(((w-self.ct.widthheight[oid][0])/w)+1)
                swy = vy_pix_per_frame*(((h-self.ct.widthheight[oid][1])/h)+1)
                if oid in self.prev_speeds:
                    v0=self.prev_speeds[oid]
                    ax=(swx**2 - v0[0]**2)/(self.T*intv)
                    ay=(swy**2 - v0[1]**2)/(self.T*intv)
                    self.accelerations.setdefault(oid,[]).append([[ax,ay],idx])
                self.prev_speeds[oid]=(swx,swy)
    def _overlaps_angles(self, idx):
        for o1 in self.centroids:
            if o1 in self.ct.objects:
                c1=self.centroids[o1][-1]; wh1=self.ct.widthheight[o1]
                for o2 in self.centroids:
                    if o2<=o1 or o2 not in self.ct.objects: continue
                    c2=self.centroids[o2][-1]; wh2=self.ct.widthheight[o2]
                    if 2*abs(c1[0]-c2[0])<wh1[0]+wh2[0] and 2*abs(c1[1]-c2[1])<wh1[1]+wh2[1]:
                        pair=(o1,o2)
                        self.overlaps.setdefault(pair,[]).append(idx)
                        if o1 in self.norm_diffs and o2 in self.norm_diffs:
                            n1=self.norm_diffs[o1][-1][0]
                            n2=self.norm_diffs[o2][-1][0]
                            dot=max(min(n1[0]*n2[0]+n1[1]*n2[1],1),-1)
                            theta=acos(dot)
                            if theta>=pi/2: theta-=pi
                            self.angles.setdefault(pair,[]).append([theta,idx])
    def get_velocity_at_frame(self, vehicle_id, target_frame, window=5):
        if vehicle_id not in self.velocities:
            return None
        velocities = self.velocities[vehicle_id]
        relevant_velocities = [v for v in velocities if abs(v['frame'] - target_frame) <= window]
        if not relevant_velocities:
            return None
        avg_vx = np.mean([v['vx'] for v in relevant_velocities])
        avg_vy = np.mean([v['vy'] for v in relevant_velocities])
        avg_speed = np.mean([v['speed'] for v in relevant_velocities])
        return {'vx': avg_vx,'vy': avg_vy,'speed': avg_speed,'direction': relevant_velocities[-1]['direction']}
    def compute_scores(self, back=15, forward=15):
        self.final_scores={}
        for pair, idxs in self.overlaps.items():
            alpha_max, beta_max, gamma_max = 0, 0, 0
            for f in idxs:
                before1=[a for a in self.accelerations.get(pair[0],[]) if a[1]<f]
                after1=[a for a in self.accelerations.get(pair[0],[]) if a[1]>=f]
                before2=[a for a in self.accelerations.get(pair[1],[]) if a[1]<f]
                after2=[a for a in self.accelerations.get(pair[1],[]) if a[1]>=f]
                if len(before1)>=back and len(before2)>=back and len(after1)>=forward and len(after2)>=forward:
                    b1=np.mean([np.hypot(*acc[0]) for acc in before1[-back:]])
                    b2=np.mean([np.hypot(*acc[0]) for acc in before2[-back:]])
                    a1=max([np.hypot(*acc[0]) for acc in after1[:forward]], default=0)
                    a2=max([np.hypot(*acc[0]) for acc in after2[:forward]], default=0)
                    diff = (a1-b1)+(a2-b2)
                    alpha = min(max((abs(diff)-100)/1400,0),1)
                    alpha_max=max(alpha_max,alpha)
            for item in self.angles.get(pair,[]):
                theta = item[0]
                beta = 0 if abs(theta)<pi/4 else abs(theta)/(pi/2)
                beta_max=max(beta_max,beta)
            diffs=[abs(self.angles[pair][i][0]-self.angles[pair][i-1][0]) for i in range(1,len(self.angles.get(pair,[])))]
            gamma_max = min(max(max(diffs, default=0)/0.8,0),1)
            final_score = 0.2*alpha_max + 0.35*beta_max + 0.45*gamma_max
            self.final_scores[pair]=final_score
        return self.final_scores
    def analyze_collision(self, vehicle_pair, collision_frame, video_fps):
        v1_id, v2_id = vehicle_pair
        v1_class = self.ct.classes.get(v1_id, {'class_name': 'unknown', 'class_id': -1})
        v2_class = self.ct.classes.get(v2_id, {'class_name': 'unknown', 'class_id': -1})
        before_frame = collision_frame - int(2 * video_fps)
        v1_before = self.get_velocity_at_frame(v1_id, before_frame)
        v2_before = self.get_velocity_at_frame(v2_id, before_frame)
        after_frame = collision_frame + int(2 * video_fps)
        v1_after = self.get_velocity_at_frame(v1_id, after_frame)
        v2_after = self.get_velocity_at_frame(v2_id, after_frame)
        collision_angle = "Unknown"
        if v1_before and v2_before:
            dot_product = (v1_before['direction'][0] * v2_before['direction'][0] + v1_before['direction'][1] * v2_before['direction'][1])
            angle_rad = acos(max(min(dot_product, 1), -1))
            angle_deg = np.degrees(angle_rad)
            if angle_deg < 30:
                collision_angle = "Rear-end"
            elif angle_deg > 150:
                collision_angle = "Head-on"
            elif 60 < angle_deg < 120:
                collision_angle = "Side-impact"
            else:
                collision_angle = f"Angled ({angle_deg:.1f}°)"
        collision_info = {'vehicle_pair': vehicle_pair,'collision_frame': collision_frame,'collision_time': collision_frame / video_fps,'collision_angle': collision_angle,'vehicle_1': {'id': v1_id,'class': v1_class,'before_collision': v1_before,'after_collision': v1_after},'vehicle_2': {'id': v2_id,'class': v2_class,'before_collision': v2_before,'after_collision': v2_after}}
        return collision_info
    def detect_accidents(self, frame_idx, scores, video_fps):
        max_score = max(scores.values()) if scores else 0
        max_pair = max(scores.items(), key=lambda x: x[1])[0] if scores else None
        self.frame_scores[frame_idx] = max_score
        if max_score >= self.accident_thresh:
            is_new_accident = True
            min_gap_frames = int(video_fps * 10)
            for prev_frame in self.accident_moments:
                if abs(frame_idx - prev_frame) < min_gap_frames:
                    is_new_accident = False
                    break
            if is_new_accident:
                self.accident_moments.append(frame_idx)
                if max_pair:
                    collision_info = self.analyze_collision(max_pair, frame_idx, video_fps)
                    self.collision_details[frame_idx] = collision_info
            return True
        return False
    def create_clip_segments(self, total_frames, video_fps, before_seconds=4, after_seconds=4):
        before_frames = int(before_seconds * video_fps)
        after_frames = int(after_seconds * video_fps)
        segments = []
        for accident_frame in self.accident_moments:
            start_frame = max(0, accident_frame - before_frames)
            end_frame = min(total_frames - 1, accident_frame + after_frames)
            segments.append((start_frame, end_frame, accident_frame))
        return segments
    def save_clip_from_video(self, input_path, output_path, start_frame, end_frame, annotated=False):
        cap = cv2.VideoCapture(input_path)
        w, h, fps = int(cap.get(3)), int(cap.get(4)), cap.get(5)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        frame_count = start_frame
        while frame_count <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            if annotated and frame_count in self.all_frames:
                frame = self.all_frames[frame_count]
            writer.write(frame)
            frame_count += 1
        cap.release()
        writer.release()
        self.make_browser_compatible(output_path)
    def run_and_save(self, inp, outp, before_seconds=4, after_seconds=4):
        cap = cv2.VideoCapture(inp)
        w, h, fps = int(cap.get(3)), int(cap.get(4)), cap.get(5)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        writer = cv2.VideoWriter(outp, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        idx = 0
        frame_number = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            idx += self.frame_inc
            processed_frame, objs, boxes = self.process_frame(frame.copy(), idx)
            scores = self.compute_scores()
            is_accident = self.detect_accidents(frame_number, scores, fps)
            high_risk_ids = set()
            annotated_frame = processed_frame.copy()
            for (o1, o2), score in scores.items():
                if score >= self.risk_thresh:
                    c1 = self.centroids[o1][-1] if o1 in self.centroids else (0, 0)
                    c2 = self.centroids[o2][-1] if o2 in self.centroids else (0, 0)
                    cv2.putText(annotated_frame, f"Risk: {score:.2f}", ((c1[0] + c2[0]) // 2, (c1[1] + c2[1]) // 2 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    if is_accident and frame_number in self.collision_details:
                        collision_info = self.collision_details[frame_number]
                        if collision_info['vehicle_pair'] == (o1, o2):
                            cv2.putText(annotated_frame, f"COLLISION: {collision_info['collision_angle']}", ((c1[0] + c2[0]) // 2, (c1[1] + c2[1]) // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    high_risk_ids.update([o1, o2])
            for oid, centroid in objs.items():
                color = (0, 0, 255) if oid in high_risk_ids else (0, 255, 0)
                if oid in self.ct.widthheight:
                    wh = self.ct.widthheight[oid]
                    top_left = (centroid[0] - wh[0] // 2, centroid[1] - wh[1] // 2)
                    bottom_right = (centroid[0] + wh[0] // 2, centroid[1] + wh[1] // 2)
                    cv2.rectangle(annotated_frame, top_left, bottom_right, color, 2)
                    class_info = self.ct.classes.get(oid, {'class_name': 'unknown', 'confidence': 0})
                    class_name = class_info['class_name'].capitalize()
                    current_velocity = self.get_velocity_at_frame(oid, idx)
                    if current_velocity:
                        label = f"{class_name} ID{oid} - {current_velocity['speed']:.1f}m/s"
                    else:
                        label = f"{class_name} ID{oid}"
                    cv2.putText(annotated_frame, f"{class_name} ID{oid}", (top_left[0], top_left[1] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    if current_velocity:
                        cv2.putText(annotated_frame, f"{current_velocity['speed']:.1f} m/s", (top_left[0], top_left[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            class_counts = {}
            for oid in objs.keys():
                if oid in self.ct.classes:
                    class_name = self.ct.classes[oid]['class_name']
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
            y_offset = 30
            for class_name, count in class_counts.items():
                cv2.putText(annotated_frame, f"{class_name.capitalize()}s: {count}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y_offset += 25
            if is_accident:
                cv2.putText(annotated_frame, "ACCIDENT DETECTED!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            self.all_frames[frame_number] = annotated_frame.copy()
            writer.write(annotated_frame)
            frame_number += 1
        cap.release()
        writer.release()
        if self.accident_moments:
            segments = self.create_clip_segments(total_frames, fps, before_seconds, after_seconds)
            base_name = os.path.splitext(outp)[0]
            self.generate_collision_report(f"{base_name}_collision_report.txt", fps)
            for i, (start_frame, end_frame, accident_frame) in enumerate(segments):
                annotated_clip_path = f"{base_name}_accident_{i+1}_annotated.mp4"
                self.save_annotated_clip(annotated_clip_path, start_frame, end_frame, fps)
                raw_clip_path = f"{base_name}_accident_{i+1}_raw.mp4"
                self.save_clip_from_video(inp, raw_clip_path, start_frame, end_frame, annotated=False)
        self.make_browser_compatible(outp)
    def generate_collision_report(self, report_path, fps):
        with open(report_path, 'w') as f:
            f.write("COLLISION DETECTION REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Analysis Summary:\n")
            f.write(f"- Total accidents detected: {len(self.accident_moments)}\n")
            f.write(f"- Video FPS: {fps}\n")
            f.write(f"- Accident threshold: {self.accident_thresh}\n")
            f.write(f"- Risk threshold: {self.risk_thresh}\n\n")
            if not self.accident_moments:
                f.write("No accidents detected in this video.\n")
                return
            for i, accident_frame in enumerate(self.accident_moments):
                f.write(f"\nACCIDENT #{i+1}\n")
                f.write("-" * 30 + "\n")
                f.write(f"Frame: {accident_frame}\n")
                f.write(f"Time: {accident_frame/fps:.2f} seconds\n")
                if accident_frame in self.collision_details:
                    details = self.collision_details[accident_frame]
                    v1 = details['vehicle_1']
                    f.write(f"\nVehicle 1 (ID {v1['id']}):\n")
                    f.write(f"  Class: {v1['class']['class_name'].capitalize()}\n")
                    f.write(f"  Confidence: {v1['class']['confidence']:.2f}\n")
                    if v1['before_collision']:
                        f.write(f"  Speed before collision: {v1['before_collision']['speed']:.2f} m/s\n")
                        f.write(f"  Velocity before: ({v1['before_collision']['vx']:.2f}, {v1['before_collision']['vy']:.2f}) m/s\n")
                    if v1['after_collision']:
                        f.write(f"  Speed after collision: {v1['after_collision']['speed']:.2f} m/s\n")
                        f.write(f"  Velocity after: ({v1['after_collision']['vx']:.2f}, {v1['after_collision']['vy']:.2f}) m/s\n")
                    v2 = details['vehicle_2']
                    f.write(f"\nVehicle 2 (ID {v2['id']}):\n")
                    f.write(f"  Class: {v2['class']['class_name'].capitalize()}\n")
                    f.write(f"  Confidence: {v2['class']['confidence']:.2f}\n")
                    if v2['before_collision']:
                        f.write(f"  Speed before collision: {v2['before_collision']['speed']:.2f} m/s\n")
                        f.write(f"  Velocity before: ({v2['before_collision']['vx']:.2f}, {v2['before_collision']['vy']:.2f}) m/s\n")
                    if v2['after_collision']:
                        f.write(f"  Speed after collision: {v2['after_collision']['speed']:.2f} m/s\n")
                        f.write(f"  Velocity after: ({v2['after_collision']['vx']:.2f}, {v2['after_collision']['vy']:.2f}) m/s\n")
                    if v1['before_collision'] and v2['before_collision']:
                        combined_speed = v1['before_collision']['speed'] + v2['before_collision']['speed']
                        f.write(f"\nImpact Analysis:\n")
                        f.write(f"  Combined pre-collision speed: {combined_speed:.2f} m/s\n")
                        if combined_speed > 20:
                            severity = "High"
                        elif combined_speed > 10:
                            severity = "Medium"
                        else:
                            severity = "Low"
                        f.write(f"  Estimated severity: {severity}\n")
                if accident_frame in self.frame_scores:
                    f.write(f"\nRisk Score: {self.frame_scores[accident_frame]:.3f}\n")
                f.write("\n" + "=" * 50 + "\n")
        print(f"📊 Collision report saved: {report_path}")
    def save_annotated_clip(self, output_path, start_frame, end_frame, video_fps):
        if not self.all_frames:
            print("No annotated frames available")
            return
        first_frame = list(self.all_frames.values())[0]
        h, w = first_frame.shape[:2]
        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), video_fps, (w, h))
        for frame_idx in range(start_frame, end_frame + 1):
            if frame_idx in self.all_frames:
                writer.write(self.all_frames[frame_idx])
            else:
                black_frame = np.zeros((h, w, 3), dtype=np.uint8)
                writer.write(black_frame)
        writer.release()
        self.make_browser_compatible(output_path)
    def make_browser_compatible(self, input_path, output_path=None):
        if output_path is None:
            output_path = input_path + '.tmp.mp4'
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"Failed to open video: {input_path}")
            return
        for fourcc_str in ['avc1', 'H264', 'mp4v']:
            fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
            fps = cap.get(cv2.CAP_PROP_FPS)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
            if out.isOpened():
                print(f"Using codec: {fourcc_str}")
                break
            out.release()
        else:
            print("No suitable codec found for browser compatibility.")
            cap.release()
            return
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
        cap.release()
        out.release()
        if output_path != input_path:
            os.replace(output_path, input_path)
    def get_structured_accident_details(self, fps):
        accidents = []
        for i, accident_frame in enumerate(self.accident_moments):
            accident = {'accident_number': int(i+1),'frame': int(accident_frame),'time': float(round(accident_frame / fps, 2)),'collision_type': None,'vehicle_1': None,'vehicle_2': None,'risk_score': float(self.frame_scores.get(accident_frame, 0)) if accident_frame in self.frame_scores else None,'impact_severity': None}
            if accident_frame in self.collision_details:
                details = self.collision_details[accident_frame]
                accident['collision_type'] = details['collision_angle']
                v1 = details['vehicle_1']
                v2 = details['vehicle_2']
                def safe_vel(vel):
                    if not vel or not isinstance(vel, dict):
                        return None
                    return {'frame': int(vel['frame']) if 'frame' in vel else None,'vx': float(vel['vx']) if 'vx' in vel else None,'vy': float(vel['vy']) if 'vy' in vel else None,'speed': float(vel['speed']) if 'speed' in vel else None,'direction': [float(x) for x in vel['direction']] if 'direction' in vel else None}
                accident['vehicle_1'] = {'id': int(v1['id']),'class': v1['class']['class_name'],'confidence': float(v1['class'].get('confidence', 0)) if v1['class'].get('confidence', None) is not None else None,'before_collision': safe_vel(v1['before_collision']),'after_collision': safe_vel(v1['after_collision'])}
                accident['vehicle_2'] = {'id': int(v2['id']),'class': v2['class']['class_name'],'confidence': float(v2['class'].get('confidence', 0)) if v2['class'].get('confidence', None) is not None else None,'before_collision': safe_vel(v2['before_collision']),'after_collision': safe_vel(v2['after_collision'])}
                if v1['before_collision'] and v2['before_collision']:
                    combined_speed = float(v1['before_collision']['speed']) + float(v2['before_collision']['speed'])
                    if combined_speed > 20:
                        severity = "High"
                    elif combined_speed > 10:
                        severity = "Medium"
                    else:
                        severity = "Low"
                    accident['impact_severity'] = severity
            accidents.append(accident)
        return accidents 