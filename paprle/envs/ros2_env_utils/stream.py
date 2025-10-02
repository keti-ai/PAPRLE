import asyncio
import json
import cv2
import numpy as np
import logging

from aiohttp import web
import aiohttp_cors
from aiohttp_cors import setup as cors_setup
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
import av

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

pcs = set()

QUEUE = asyncio.Queue()

class RealSenseMultiCamTrack(VideoStreamTrack):
    def __init__(self):
        super().__init__()
        self.last_frame = None

    async def recv(self):
        pts, time_base = await self.next_timestamp()

        frame = await QUEUE.get()
        if frame is None:
            frame = self.last_frame
        else:
            self.last_frame = frame

        return frame

    def stop(self):
        self.queue = None

async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pcs.add(pc)

    track = RealSenseMultiCamTrack()
    pc.addTrack(track)

    @pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange():
        logger.info(f"ICE connection state: {pc.iceConnectionState}")
        if pc.iceConnectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info(f"Connection state: {pc.connectionState}")
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    @pc.on("track")
    def on_track(track):
        logger.info(f"Track received: {track.kind}")

    try:
        await pc.setRemoteDescription(offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        return web.Response(
            content_type="application/json",
            text=json.dumps({
                "sdp": pc.localDescription.sdp,
                "type": pc.localDescription.type
            })
        )
    except Exception as e:
        logger.error(f"Error in offer: {e}")
        await pc.close()
        pcs.discard(pc)
        return web.Response(status=500, text=str(e))

async def health_check(request):
    """Health check endpoint"""
    return web.Response(text="OK")

async def index(request):
    """Serve the test HTML page"""
    with open('test_client.html', 'r') as f:
        content = f.read()
    return web.Response(text=content, content_type='text/html')

async def on_shutdown(app):
    """Clean up on shutdown"""
    logger.info("Shutting down WebRTC server...")
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()

def run_image_server(queue):

    app = web.Application()
    app.on_shutdown.append(on_shutdown)

    # Setup CORS properly
    cors = cors_setup(app, defaults={
        "*": aiohttp_cors.ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_headers="*",
            allow_methods="*"
        )
    })

    # Add routes
    app.router.add_post("/offer", offer)
    app.router.add_get("/health", health_check)
    app.router.add_get("/", index)

    # Add CORS to all routes
    for route in list(app.router.routes()):
        cors.add(route)

    web.run_app(app, port=8080, host="0.0.0.0")
