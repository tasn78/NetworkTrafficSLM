from scapy.all import send
from scapy.layers.inet import IP, TCP, UDP
from faker import Faker
import random
import threading
import time
import socket
import os

fake = Faker()

def simulate_dos(target_ip, target_port, packet_count=10):
    print(f"Simulating DDoS attack on {target_ip}:{target_port}")
    for i in range(packet_count):
        src_ip = fake.ipv4()
        src_port = random.randint(1024, 65535)
        packet = IP(src=src_ip, dst=target_ip) / TCP(sport=src_port, dport=target_port, flags="S")
        print(f"Packet {i + 1}: {packet.summary()}")
        send(packet, verbose=True)


def simulate_port_scan(target_ip, start_port, end_port):
    """
    Simulate a port scan on the target IP from a range of ports.
    """
    print(f"Simulating port scan on {target_ip} from port {start_port} to {end_port}...")
    for port in range(start_port, end_port + 1):
        src_port = fake.random_int(min=1024, max=65535)
        packet = IP(src=fake.ipv4(), dst=target_ip) / TCP(sport=src_port, dport=port, flags="S")
        send(packet, verbose=False)

def generate_benign_traffic(target_ip, target_port, duration=30):
    """
    Generate benign traffic by sending random packets to the target.
    """
    print(f"Generating benign traffic to {target_ip}:{target_port} for {duration} seconds...")
    end_time = time.time() + duration
    while time.time() < end_time:
        packet = IP(src=fake.ipv4(), dst=target_ip) / TCP(sport=random.randint(1024, 65535), dport=target_port, flags="A")
        send(packet, verbose=False)
        time.sleep(0.5)  # Add delay between benign packets

def get_local_ip():
    """
    Automatically fetch the local IP address of your device.
    """
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    print(f"Targeting your local IP address: {local_ip}")
    return local_ip

def attack_simulation(target_ip, target_port, attack_type, benign_traffic_duration=30):
    """
    Simulate selected attack type and generate benign traffic in parallel.
    """
    # Generate benign traffic in a background thread
    benign_thread = threading.Thread(target=generate_benign_traffic, args=(target_ip, target_port, benign_traffic_duration))
    benign_thread.start()

    # Perform the selected attack
    if attack_type == "ddos":
        simulate_dos(target_ip, target_port, packet_count=100)
    elif attack_type == "port_scan":
        simulate_port_scan(target_ip, start_port=target_port, end_port=target_port + 10)
    else:
        print(f"Unknown attack type: {attack_type}")

    benign_thread.join()  # Wait for the benign traffic to finish

def main():
    # Get target IP and port
    target_ip = get_local_ip()
    target_port = int(input("Enter the target port (e.g., 80 or 443): "))
    attack_type = input("Enter the type of attack (ddos, port_scan): ").lower()

    # Simulate the attack
    attack_simulation(target_ip, target_port, attack_type)

if __name__ == "__main__":
    main()
