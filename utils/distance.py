foc = 1.6        # Lens focal length, in cm
real_hight_bicycle = 26.04      # The height of the bike, note that the units are inches
real_hight_car = 59.08      # Car height
real_hight_motorcycle = 47.24      # Motorcycle height
real_hight_bus = 125.98      # Bus height
real_hight_truck = 137.79   # Truck height
ppi=400

# Custom function, single visual distance measurement
def detect_distance_car(h):
    dis_inch = (real_hight_car * foc) * ppi / (h - 4)
    dis_cm = dis_inch * 2.54
    dis_cm = int(dis_cm)
    dis_m = dis_cm/100
    #print(h)
    return dis_m

def detect_distance_bicycle(h):
    dis_inch = (real_hight_bicycle * foc) * ppi / (h - 4)
    dis_cm = dis_inch * 2.54
    dis_cm = int(dis_cm)
    dis_m = dis_cm / 100
    #print(h)
    return dis_m

def detect_distance_motorcycle(h):
    dis_inch = (real_hight_motorcycle * foc) * ppi / (h - 4)
    dis_cm = dis_inch * 2.54
    dis_cm = int(dis_cm)
    dis_m = dis_cm/100
    #print(h)
    return dis_m

def detect_distance_bus(h):
    dis_inch = (real_hight_bus * foc) * ppi / (h - 4)
    dis_cm = dis_inch * 2.54
    dis_cm = int(dis_cm)
    dis_m = dis_cm/100
    #print(h)
    return dis_m

def detect_distance_truck(h):
    dis_inch = (real_hight_truck * foc) * ppi / (h - 4)
    dis_cm = dis_inch * 2.54
    dis_cm = int(dis_cm)
    dis_m = dis_cm/100
    #print(h)
    return dis_m








