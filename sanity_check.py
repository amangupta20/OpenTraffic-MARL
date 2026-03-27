import libsumo
import pathlib

cfg = str(pathlib.Path('/app/sumo_net/bangalore_mg_road/osm.sumocfg'))
libsumo.start([
    'sumo', '-c', cfg,
    '--no-step-log', 'true',
    '--no-warnings', 'true',
    '--ignore-route-errors', 'true',
    '--additional-files', '/app/sumo_net/bangalore_mg_road/osm.poly.xml.gz,/app/sumo_net/bangalore_mg_road/osm_stops.add.xml',
    '--tripinfo-output', '/tmp/tripinfo.xml',
    '--stop-output', '/tmp/stopinfo.xml',
    '--emission-output', '/tmp/emission.xml',
    '--statistic-output', '/tmp/stats.xml',
    '--scale', '1.0',
    '--seed', '42',
])

arrived = 0
teleported = 0
max_queue = 0
lane_ids = [lid for lid in libsumo.lane.getIDList() if not lid.startswith(':')]

print(f"Network loaded. Controlled lanes: {len(lane_ids)}")
print(f"TLS junctions: {libsumo.trafficlight.getIDList()}")
print()
print(f"{'Step':>6}  {'In_Sim':>6}  {'Halting':>7}  {'Wait(s)':>9}  {'Arrived':>8}  {'Teleport':>9}")
print("-" * 60)

for step in range(1800):
    libsumo.simulationStep()
    arrived += libsumo.simulation.getArrivedNumber()
    teleported += libsumo.simulation.getEndingTeleportNumber()

    if step % 120 == 0:
        in_sim = libsumo.simulation.getMinExpectedNumber()
        halting = sum(libsumo.lane.getLastStepHaltingNumber(lid) for lid in lane_ids)
        waiting = sum(libsumo.lane.getWaitingTime(lid) for lid in lane_ids)
        max_queue = max(max_queue, halting)
        print(f"{step:6d}  {in_sim:6d}  {halting:7d}  {waiting:9.0f}  {arrived:8d}  {teleported:9d}")

print()
total = arrived + teleported
print("=" * 60)
print(f"SUMMARY — 1800 simulation seconds at scale=1.0")
print(f"  Total arrived:     {arrived}")
print(f"  Total teleported:  {teleported}")
print(f"  Peak queue:        {max_queue} halting vehicles")
print(f"  Teleport rate:     {teleported/total*100:.1f}%" if total > 0 else "  No vehicles")
print()

# Now check scale=0.5
libsumo.close()

print("Re-running at scale=0.5 for 600s...")
libsumo.start([
    'sumo', '-c', cfg,
    '--no-step-log', 'true',
    '--no-warnings', 'true',
    '--ignore-route-errors', 'true',
    '--additional-files', '/app/sumo_net/bangalore_mg_road/osm.poly.xml.gz,/app/sumo_net/bangalore_mg_road/osm_stops.add.xml',
    '--tripinfo-output', '/tmp/tripinfo2.xml',
    '--stop-output', '/tmp/stopinfo2.xml',
    '--emission-output', '/tmp/emission2.xml',
    '--statistic-output', '/tmp/stats2.xml',
    '--scale', '0.5',
    '--seed', '42',
])

arrived2 = teleported2 = max_queue2 = 0
for step in range(600):
    libsumo.simulationStep()
    arrived2 += libsumo.simulation.getArrivedNumber()
    teleported2 += libsumo.simulation.getEndingTeleportNumber()
    if step % 120 == 0:
        in_sim = libsumo.simulation.getMinExpectedNumber()
        halting = sum(libsumo.lane.getLastStepHaltingNumber(lid) for lid in lane_ids)
        max_queue2 = max(max_queue2, halting)
        print(f"  scale=0.5 t={step:4d}s  in_sim={in_sim}  halting={halting}  arrived={arrived2}  teleport={teleported2}")

libsumo.close()
print(f"\nscale=0.5 peak queue: {max_queue2}")
print("Done.")
