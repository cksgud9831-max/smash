using System;
using System.Net.Sockets;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;

/// <summary>
/// 유니티 환경의 카메라 프레임과 센서 데이터를 Python 조준 엔진 서버로 전송하고, 
/// 조준 결과를 피드백으로 수신받는 시뮬레이션 클라이언트입니다.
/// </summary>
public class AimingSimulationClient : MonoBehaviour
{
    [Header("Network Settings")]
    public string serverIP = "127.0.0.1";
    public int serverPort = 8080;
    
    [Header("Scope Setup")]
    public Camera scopeCamera;
    public RenderTexture renderTexture;
    public float raycastMaxDistance = 1000f;
    [Range(0, 100)] public int jpgQuality = 70;

    private TcpClient client;
    private NetworkStream stream;
    private bool isConnected = false;
    private Texture2D tex2D;

    // 파이썬으로 보낼 JSON 구조
    [Serializable]
    public class SensorPayload {
        public float timestamp;
        public float distance;
        public float[] pose;
        public int image_size;
    }

    // 파이썬에서 받을 JSON 구조
    [Serializable]
    public class FeedbackPayload {
        public string state;
        public float aim_azimuth;
        public float aim_elevation;
        public bool aim_ready;
    }

    void Start()
    {
        if (scopeCamera == null) scopeCamera = Camera.main;
        
        // 렌더 텍스처를 읽어오기 위한 Texture2D 메모리 할당
        tex2D = new Texture2D(renderTexture.width, renderTexture.height, TextureFormat.RGB24, false);
        
        ConnectToServer();
    }

    async void ConnectToServer()
    {
        try {
            client = new TcpClient();
            await client.ConnectAsync(serverIP, serverPort);
            stream = client.GetStream();
            isConnected = true;
            Debug.Log("[AimingClient] 파이썬 서버에 연결되었습니다.");
            
            // 파이썬으로부터 조준 피드백을 수신하는 백그라운드 루프 시작
            _ = ReceiveFeedbackLoopAsync();
        } catch (Exception e) {
            Debug.LogError($"[AimingClient] 서버 연결 실패 (서버가 켜져 있는지 확인하세요): {e.Message}");
        }
    }

    // 모든 렌더링이 끝난 직후 캡처하기 위해 LateUpdate 사용
    void LateUpdate()
    {
        if (!isConnected || stream == null) return;
        
        // 1. 카메라 화면 캡처 및 JPG 인코딩
        RenderTexture.active = renderTexture;
        tex2D.ReadPixels(new Rect(0, 0, renderTexture.width, renderTexture.height), 0, 0);
        tex2D.Apply();
        RenderTexture.active = null;
        
        byte[] imageBytes = tex2D.EncodeToJPG(jpgQuality);

        // 2. 가상 레이저 측거 (카메라 중앙에서 앞쪽으로 Raycast)
        float distance = -1f;
        Ray ray = new Ray(scopeCamera.transform.position, scopeCamera.transform.forward);
        if (Physics.Raycast(ray, out RaycastHit hit, raycastMaxDistance)) {
            distance = hit.distance;
        }

        // 3. 카메라 자세(Transform) 4x4 행렬 추출
        Matrix4x4 mat = scopeCamera.transform.localToWorldMatrix;
        float[] poseArray = new float[16] {
            mat.m00, mat.m01, mat.m02, mat.m03,
            mat.m10, mat.m11, mat.m12, mat.m13,
            mat.m20, mat.m21, mat.m22, mat.m23,
            mat.m30, mat.m31, mat.m32, mat.m33
        };

        // 4. 데이터 패키징 (JSON)
        SensorPayload payload = new SensorPayload {
            timestamp = Time.time,
            distance = distance,
            pose = poseArray,
            image_size = imageBytes.Length
        };

        string jsonStr = JsonUtility.ToJson(payload);
        byte[] jsonBytes = Encoding.UTF8.GetBytes(jsonStr);

        // 5. 프로토콜 규칙에 맞게 전송 (4바이트 길이 정보 -> JSON -> 이미지 바이트)
        byte[] lengthBytes = BitConverter.GetBytes((uint)jsonBytes.Length);
        if (!BitConverter.IsLittleEndian) Array.Reverse(lengthBytes); // 리틀 엔디안 보장
        
        _ = SendDataAsync(lengthBytes, jsonBytes, imageBytes);
    }

    async Task SendDataAsync(byte[] len, byte[] json, byte[] img)
    {
        try {
            await stream.WriteAsync(len, 0, len.Length);
            await stream.WriteAsync(json, 0, json.Length);
            if (img.Length > 0) {
                await stream.WriteAsync(img, 0, img.Length);
            }
        } catch (Exception e) {
            Debug.LogWarning($"[AimingClient] 데이터 전송 오류: {e.Message}");
            isConnected = false;
        }
    }

    async Task ReceiveFeedbackLoopAsync()
    {
        byte[] lenBuf = new byte[4];
        while (isConnected) {
            try {
                // 1. 파이썬이 보내는 JSON 바이트 길이(4 byte) 수신
                int read = await stream.ReadAsync(lenBuf, 0, 4);
                if (read == 0) break;
                
                if (!BitConverter.IsLittleEndian) Array.Reverse(lenBuf);
                int jsonLen = BitConverter.ToInt32(lenBuf, 0);

                // 2. JSON 문자열 수신
                byte[] jsonBuf = new byte[jsonLen];
                read = await stream.ReadAsync(jsonBuf, 0, jsonLen);
                if (read == 0) break;

                string jsonStr = Encoding.UTF8.GetString(jsonBuf);
                FeedbackPayload feedback = JsonUtility.FromJson<FeedbackPayload>(jsonStr);
                
                // TODO: 여기서 feedback 데이터를 바탕으로 유니티 상의 십자선(HUD) 위치를 
                // 조준 보정값(aim_azimuth, aim_elevation)만큼 이동시키는 UI 로직을 추가하시면 됩니다.
                // Debug.Log($"조준 상태: {feedback.state} | 발사 준비: {feedback.aim_ready}");

            } catch (Exception) {
                break;
            }
        }
        isConnected = false;
        Debug.Log("[AimingClient] 서버와의 연결이 종료되었습니다.");
    }

    void OnDestroy()
    {
        isConnected = false;
        if (stream != null) stream.Close();
        if (client != null) client.Close();
    }
}
