import psutil
from scapy.all import sniff


def get_active_interface():
    # Get network interfaces and check if they are connected
    interfaces = psutil.net_if_stats()
    for iface_name, iface_info in interfaces.items():
        if iface_info.isup:  # Check if interface is active
            if "Wi-Fi" in iface_name or "Ethernet" in iface_name:
                return iface_name
    return None


def process_packet(packet):
    # Print packet details and the interface it was captured on
    print(f"Packet received on interface: {packet.sniffed_on}")
    packet.show()  # Display detailed information about the packet


def start_monitoring():
    interface = get_active_interface()
    if interface:
        print(f"Starting network monitoring on {interface}")
        sniff(iface=interface, prn=process_packet, filter="tcp")
    else:
        print("No active interface found. Make sure you are connected to Wi-Fi or Ethernet.")


if __name__ == "__main__":
    start_monitoring()
