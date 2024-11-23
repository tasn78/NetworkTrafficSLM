import psutil
import socket
from scapy.all import sniff

def get_active_interface():
    # Get network interfaces and their stats
    interfaces = psutil.net_if_addrs()
    io_counters = psutil.net_io_counters(pernic=True)

    for iface_name, iface_addrs in interfaces.items():
        for addr in iface_addrs:
            # Check for an IPv4 address
            if addr.family == socket.AF_INET:
                # Ensure the interface is actively sending or receiving data
                if iface_name in io_counters:
                    stats = io_counters[iface_name]
                    if stats.bytes_sent > 0 or stats.bytes_recv > 0:  # Check for traffic
                        if "Wi-Fi" in iface_name or "Ethernet" in iface_name:
                            return iface_name
    return None

def process_packet(packet):
    try:
        print(f"Packet received on interface: {packet.sniffed_on}")
        if packet.haslayer("TCP"):
            print(f"Source IP: {packet[IP].src}, Destination IP: {packet[IP].dst}")
            print(f"Source Port: {packet[TCP].sport}, Destination Port: {packet[TCP].dport}")

            # Check for suspicious ports or IPs
            if packet[TCP].dport not in [80, 443]:  # Alert if not HTTP/HTTPS
                print("⚠️ Suspicious Port Detected!")
            if "192.168." not in packet[IP].src and "192.168." not in packet[IP].dst:
                print("⚠️ External Traffic Detected!")

        packet.show()  # Show full details of the packet
    except Exception as e:
        print(f"Error processing packet: {e}")


def start_monitoring():
    interface = get_active_interface()
    if interface:
        print(f"Starting network monitoring on {interface}")
        sniff(iface=interface, prn=process_packet, filter="tcp")
    else:
        print("No active interface found. Make sure you are connected to Wi-Fi or Ethernet.")

if __name__ == "__main__":
    start_monitoring()
