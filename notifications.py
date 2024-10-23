# notifications.py

def send_alert(packet, details):
    print(f"ALERT: Suspicious traffic detected!\nDetails: {details}\nPacket summary: {packet.summary()}")
