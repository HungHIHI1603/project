import time
from datetime import datetime, time as datetime_time

import cv2
import streamlit as st
from ultralytics import YOLO


# =========================================================
# HOSPITALAI - GIAM SAT BENH NHAN
# =========================================================

st.set_page_config(
    page_title="HospitalAI",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 HospitalAI - Giám sát bệnh nhân")
st.caption("Theo dõi tư thế từ webcam laptop theo thời gian thực")


# =========================================================
# KHOI TAO TRANG THAI
# =========================================================

defaults = {
    "camera_on": False,
    "camera_capture": None,
    "lying_since": None,
    "alert_active": False,
    "alert_history": [],
    "last_alert_time": None,
    "last_alert_room": None,
    "last_inference_time": 0.0,
    "last_annotated_frame": None,
    "last_lying_result": False,
    "last_detection_error": None,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =========================================================
# MO MO HINH AI
# =========================================================

@st.cache_resource
def load_model():
    return YOLO("yolo11n-pose.pt")


# =========================================================
# NHAN DIEN TU THE NAM
# =========================================================

def detect_lying(result):
    if result.keypoints is None:
        return False

    points_all = result.keypoints.xy.cpu().numpy()

    if len(points_all) == 0:
        return False

    points = points_all[0]

    if len(points) < 13:
        return False

    if result.keypoints.conf is not None:
        confidence = result.keypoints.conf.cpu().numpy()[0]

        for index in [5, 6, 11, 12]:
            if confidence[index] < 0.30:
                return False

    # Trung diem hai vai
    shoulder_x = (points[5][0] + points[6][0]) / 2
    shoulder_y = (points[5][1] + points[6][1]) / 2

    # Trung diem hai hong
    hip_x = (points[11][0] + points[12][0]) / 2
    hip_y = (points[11][1] + points[12][1]) / 2

    dx = hip_x - shoulder_x
    dy = hip_y - shoulder_y

    distance = (dx * dx + dy * dy) ** 0.5

    if distance < 20:
        return False

    # Uoc luong than nguoi gan nam ngang
    return abs(dy) / distance < 0.35


# =========================================================
# CAI DAT
# =========================================================

st.sidebar.header("⚙️ Cài đặt giám sát")

room_name = st.sidebar.text_input(
    "Tên phòng",
    value="Phòng 101"
)

camera_index = st.sidebar.selectbox(
    "Webcam laptop",
    [0, 1, 2, 3],
    index=0
)

st.sidebar.divider()
st.sidebar.subheader("🕒 1. Giờ bắt đầu kiểm tra")

mode = st.sidebar.radio(
    "Chọn chế độ",
    ["Chạy thử ngay", "Theo giờ cài đặt"]
)

scheduled_time = st.sidebar.time_input(
    "Giờ bắt đầu (00:00–23:59)",
    value=datetime_time(12, 10),
    step=60
)

st.sidebar.caption(
    "Chế độ theo giờ sẽ bắt đầu kiểm tra khi đồng hồ đến giờ đã chọn."
)

st.sidebar.divider()
st.sidebar.subheader("⏱️ 2. Thời gian nằm trước khi cảnh báo")

duration_value = st.sidebar.number_input(
    "Số giây nằm liên tục",
    min_value=1,
    max_value=86400,
    value=2,
    step=1,
    help="Nhập 2 để thử nhanh, 30 để thử 30 giây, "
         "300 để thử 5 phút hoặc 3600 để thử 1 giờ."
)

st.sidebar.caption(
    f"Thời gian hiện tại: {duration_value} giây "
    f"({duration_value / 60:.2f} phút)"
)

st.sidebar.divider()
st.sidebar.subheader("📷 Độ mượt camera")

inference_interval = st.sidebar.select_slider(
    "Tần suất phân tích AI",
    options=[0.3, 0.5, 1.0, 1.5],
    value=0.5,
    format_func=lambda x: f"Mỗi {x} giây"
)

st.sidebar.caption(
    "Nếu hình ảnh giật, hãy chọn 1 hoặc 1.5 giây "
    "để giảm tải cho laptop."
)


# =========================================================
# DIEU KHIEN CAMERA
# =========================================================

col1, col2 = st.columns(2)

with col1:
    if st.button("▶️ BẬT CAMERA", use_container_width=True):
        st.session_state.camera_on = True
        st.session_state.lying_since = None
        st.session_state.last_annotated_frame = None
        st.session_state.last_lying_result = False
        st.session_state.last_detection_error = None

with col2:
    if st.button("⏹️ DỪNG CAMERA", use_container_width=True):
        st.session_state.camera_on = False

        cap = st.session_state.camera_capture

        if cap is not None:
            cap.release()

        st.session_state.camera_capture = None
        st.session_state.lying_since = None
        st.session_state.last_annotated_frame = None


# =========================================================
# THONG TIN TONG QUAN
# =========================================================

c1, c2, c3 = st.columns(3)

c1.metric("Phòng", room_name)

c2.metric(
    "Thời gian nằm yêu cầu",
    f"{duration_value} giây"
)

c3.metric(
    "Trạng thái",
    "Đang bật" if st.session_state.camera_on else "Đang tắt"
)

st.divider()


# =========================================================
# CAMERA VA AI
# =========================================================

@st.fragment(run_every=0.2)
def monitor_camera():
    st.subheader("📹 Camera trực tiếp")

    if not st.session_state.camera_on:
        st.info("Nhấn BẬT CAMERA để bắt đầu.")
        return

    now = datetime.now()

    # Kiem tra lich
    if mode == "Theo giờ cài đặt" and now.time() < scheduled_time:
        st.session_state.lying_since = None

        st.warning(
            f"Đang chờ đến {scheduled_time.strftime('%H:%M')}. "
            "Camera chưa phân tích tư thế."
        )

    else:
        st.success("Đang trong thời gian giám sát.")

    # Mo camera mot lan va giu ket noi
    cap = st.session_state.camera_capture

    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(camera_index)

        if not cap.isOpened():
            st.error(
                f"Không mở được webcam số {camera_index}. "
                "Hãy thử số 0 hoặc đóng ứng dụng đang sử dụng camera."
            )
            return

        # Giam bo dem khung hinh de uu tien hinh moi
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Giam kich thuoc xu ly de laptop nhe hon
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        st.session_state.camera_capture = cap

    success, frame = cap.read()

    if not success:
        st.warning("Chưa đọc được hình ảnh từ webcam.")
        return

    current_time = time.time()
    monitoring_allowed = (
        mode == "Chạy thử ngay"
        or now.time() >= scheduled_time
    )

    # Chi chay AI theo tan suat duoc chon
    should_infer = (
        monitoring_allowed
        and (
            current_time - st.session_state.last_inference_time
            >= inference_interval
            or st.session_state.last_annotated_frame is None
        )
    )

    if should_infer:
        try:
            model = load_model()

            results = model.predict(
                source=frame,
                conf=0.35,
                imgsz=416,
                verbose=False
            )

            result = results[0]

            st.session_state.last_annotated_frame = result.plot()
            st.session_state.last_lying_result = detect_lying(result)
            st.session_state.last_inference_time = current_time
            st.session_state.last_detection_error = None

        except Exception as error:
            st.session_state.last_detection_error = str(error)

    # Hien thi khung hinh moi nhat da phan tich
    annotated = st.session_state.last_annotated_frame

    if annotated is not None:
        st.image(
            cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
            channels="RGB",
            use_container_width=True
        )
    else:
        st.image(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            channels="RGB",
            use_container_width=True
        )

    if st.session_state.last_detection_error:
        st.error("Lỗi phân tích AI:")
        st.code(st.session_state.last_detection_error)
        return

    if not monitoring_allowed:
        return

    # Theo doi tu the nam
    if st.session_state.last_lying_result:
        st.warning("⚠️ AI nghi ngờ người đang nằm.")

        if st.session_state.lying_since is None:
            st.session_state.lying_since = time.time()

        elapsed = time.time() - st.session_state.lying_since
        remaining = max(0, duration_value - elapsed)

        st.write(
            f"Thời gian nằm: {elapsed:.1f} / "
            f"{duration_value} giây"
        )

        st.progress(min(1.0, elapsed / duration_value))

        if elapsed >= duration_value:
            if not st.session_state.alert_active:
                alert_time = datetime.now().strftime("%H:%M:%S")

                st.session_state.alert_active = True
                st.session_state.last_alert_time = alert_time
                st.session_state.last_alert_room = room_name

                st.session_state.alert_history.insert(
                    0,
                    {
                        "Phòng": room_name,
                        "Thời điểm": alert_time,
                        "Thời gian cấu hình (giây)": duration_value,
                        "Nội dung": "Nghi ngờ tư thế nằm kéo dài"
                    }
                )

    else:
        st.info(
            "AI chưa xác nhận tư thế nằm theo quy tắc hiện tại."
        )

        st.session_state.lying_since = None

        st.write("Thời gian nằm: 0 giây")
        st.progress(0.0)

    # Canh bao
    if st.session_state.alert_active:
        st.error(
            f"🚨 CẢNH BÁO: {st.session_state.last_alert_room}\n\n"
            f"Thời điểm: {st.session_state.last_alert_time}\n\n"
            "Vui lòng kiểm tra trực tiếp người đang được giám sát."
        )

        if st.button(
            "✅ XÁC NHẬN ĐÃ TIẾP NHẬN",
            key="acknowledge_alert"
        ):
            st.session_state.alert_active = False
            st.session_state.lying_since = None
            st.rerun()


monitor_camera()


# =========================================================
# LICH SU
# =========================================================

st.divider()
st.subheader("🚨 Danh sách cảnh báo")

if st.session_state.alert_history:
    st.dataframe(
        st.session_state.alert_history,
        use_container_width=True,
        hide_index=True
    )
else:
    st.success("Chưa có cảnh báo.")


# =========================================================
# GIOI HAN
# =========================================================

with st.expander("Thông tin và giới hạn"):
    st.write(
        """
        - Giờ bắt đầu và thời gian nằm là hai cài đặt riêng.
        - Thời gian nằm được nhập bằng giây, từ 1 đến 86400 giây.
        - Webcam được giữ kết nối giữa các lần cập nhật giao diện.
        - AI ước lượng tư thế bằng YOLO Pose; kết quả có thể sai.
        - Góc camera và việc cơ thể bị che khuất ảnh hưởng đến kết quả.
        - Chưa có mô hình chuyên biệt phát hiện chảy máu hoặc té ngã.
        - Cảnh báo hiện chỉ hiển thị trên giao diện, chưa gửi đến máy chủ.
        - Đây là nguyên mẫu đồ án, không thay thế thiết bị y tế.
        """
    )