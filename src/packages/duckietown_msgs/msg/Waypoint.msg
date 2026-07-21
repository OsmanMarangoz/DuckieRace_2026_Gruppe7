std_msgs/Header header
float64 x              # Target X Position in Metern
float64 y              # Target Y Position in Metern
float64 theta          # Target Orientierung in Radians
int32 tunnel_id        # Tunnel-ID wenn am Ziel ein Tunnel durchfahren werden soll (-1 = kein Tunnel)
int32 sequence_number  # Reihenfolge im Pfad (0, 1, 2, ...)
