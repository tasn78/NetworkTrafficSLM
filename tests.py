from scapy.all import sniff, get_if_list

def print_packet(packet):
    print(f"Packet received on interface: {packet.sniffed_on}")

print("Monitoring traffic on available interfaces...")
for iface in get_if_list():
    print(f"Monitoring interface: {iface}")
    sniff(iface=iface, prn=print_packet, store=False, count=10)
