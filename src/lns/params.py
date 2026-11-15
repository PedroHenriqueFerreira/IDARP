''' LNS tuning constants. '''

EPSILON = 1e-9

# Adaptive search
LEARNING_RATE = 0.3  # how fast a neighborhood type's weight follows its results
MIN_WEIGHT = 0.05  # every type keeps a chance of being chosen
MIN_SIZE = 1  # routes in the model: smallest...
START_SIZE = 2  # ...and initial
FAST = 0.3  # a solve ending within this fraction of the call time lets the size grow
MIN_REMAINING = 0.5  # seconds: below this a solver call is not worth it
RR_SHARE = 1.0  # fraction of the LNS time spent in ruin and recreate before CP-SAT

# Neighborhoods
K_NEAREST = 5  # related requests added to the cluster of each request of a route
K_OUTSIDE = 5  # carrier / bus requests added to the cluster of each route
MAX_IDLE = 2  # idle vehicles (each seeded with an outside request) per neighborhood

# Type: (weight of the time-window difference in the relatedness, outside requests chosen by savings)
NEIGHBORHOODS = {
    'spatial': (0.1, False),
    'temporal': (2.0, False),
    'savings': (0.5, True),
    'savings-temporal': (2.0, True),
}

# Ruin and recreate
RR_MIN_REMOVE = 2  # requests removed per iteration: between this and
RR_MAX_REMOVE = 30  # ... min(this, RR_MAX_FRACTION * requests)
RR_MAX_FRACTION = 0.3
RR_RANDOMNESS = 6.0  # Shaw removal: larger = sticks closer to the most related requests
RR_TEMPERATURE = 0.1  # initial temperature, as a fraction of the average cost per request
RR_COOLING = 0.01  # final temperature / initial temperature
RR_MEMO_LIMIT = 300_000  # cached insertions / removals before the cache is dropped
