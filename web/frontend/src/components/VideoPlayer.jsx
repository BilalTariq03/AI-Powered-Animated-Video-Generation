import { useRef } from 'react'

const VIDEO_URL    = '/api/outputs/video'
const DOWNLOAD_URL = '/api/outputs/video'

export default function VideoPlayer() {
  const videoRef = useRef(null)

  return (
    <div className="video-player">
      <video
        ref={videoRef}
        src={VIDEO_URL}
        controls
        className="video-el"
      />
      <div className="video-actions">
        <a
          href={DOWNLOAD_URL}
          download="final_video.mp4"
          className="btn-download"
        >
          Download MP4
        </a>
      </div>
    </div>
  )
}
