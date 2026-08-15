# Feature brainstorm — what was considered, what was tested, what survived

Selection rule throughout: a candidate is judged on the **mean of five rolling
cutoffs**, never on a single fold. Seed noise is 0.01–0.05 MAPE points
(measured over five seeds), so anything inside that band is not a result.

```bash
python feature_lab/ablation.py            # screen each group
python feature_lab/where_the_error_is.py  # ceiling, error budget, level ensemble
```

## The full brainstorm

Ideas are grouped by the reasoning that motivated them, with the ones actually
built marked.

### Freight economics

| Idea | Built | Rationale |
|---|---|---|
| **Headhaul / backhaul imbalance** | yes | A city shipping out more than it takes in is short of trucks, so rates run high. The oldest real driver of freight pricing. |
| **Corridor imbalance** | yes | Same logic per lane: loads A→B against loads B→A. Measured asymmetry in this data is 2.9%. |
| **Origin / destination volume** | yes | Thick markets price tighter than thin ones. |
| **Capacity utilisation** | yes | A 45,000 lb trailer moving 18,000 lb is a partial load and prices differently. |
| Carrier availability / truck counts | no | Not in the data. |
| Fuel surcharge | no | Not in the data, and normally billed separately from linehaul. |

### Geography

| Idea | Built | Rationale |
|---|---|---|
| **Bearing (direction of travel)** | yes | Lanes running out of a production region price differently from lanes running back. Encoded cyclically. |
| **Region clusters (k-means, k=8) with pair encoding** | yes | 12.2% of validation lanes are unseen and currently fall all the way back to the global prior. A region pair is a far better fallback. |
| **Midpoint coordinates, lat/lon deltas** | yes | Lets the tree carve corridors rather than individual cities. |
| Distance to nearest major market | no | Subsumed by region clusters. |
| Road-network distance | no | No routing data; circuity is already a feature. |

### Time and history

| Idea | Built | Rationale |
|---|---|---|
| **Recent lane price (causal expanding mean)** | yes | The static lane encoding averages ten months. If a lane drifted, that average is stale in a way the model cannot see. |
| **Month position / month-end push** | yes | Shippers push freight at month end to hit quotas. |
| Lagged market index (7/14/56 day) | no | 28-day already scores near zero on permutation importance. |
| Day-of-year Fourier terms | no | Tested in `approach2/` and unusable: an annual cycle observed less than once. |

### Model structure

| Idea | Built | Rationale |
|---|---|---|
| **Ensemble of level projections** | yes | The three methods fail in *different* regimes — peaks, turns, trends — so averaging should be steadier than any one. |
| Quantile models for a price band | no | Genuinely useful for a pricing desk, but it is a different deliverable, not an accuracy improvement. |
| Per-equipment models | no | Only three classes, and equipment is already a native categorical split. |
| Monotonic constraints | no | Worth trying only if the ablation had shown feature headroom. It did not. |

---

## Result: every feature group is inside the noise band

Mean MAPE across five cutoffs:

| Variant | mean | delta | verdict |
|---|---|---|---|
| + recent lane price | 3.633 | −0.008 | noise |
| + geometry (bearing, corridor) | 3.636 | −0.005 | noise |
| + region clusters | 3.639 | −0.002 | noise |
| **current model** | **3.641** | — | — |
| + flow (headhaul/backhaul) | 3.645 | +0.005 | noise |
| + capacity utilisation | 3.646 | +0.006 | noise |
| + calendar (month-end) | 3.682 | +0.042 | noise |

Not one group clears seed noise. That is a real finding, not a failure to try
hard enough, and it is worth stating plainly rather than shipping six features
that do nothing.

**Why.** The domain ideas above are all real in actual freight markets. They
are not present in *this* dataset. Rate here is generated from distance,
equipment, weight, lane and a drifting market level, plus noise. There is no
headhaul economics to find, no bearing effect, no month-end push. Adding a
feature for a mechanism the generator never applied can only add variance.

The two ideas that came closest — recent lane price and bearing — are also the
two whose real-world versions are strongest. They land at −0.008 and −0.005.

## Where the error actually is

See `figures/03_where_the_error_is.png`. Scoring on a random split hands the
model effectively perfect knowledge of the market level, which isolates what
the *features* can and cannot explain. Adding every candidate feature to that
ceiling moves it by less than seed noise.

So the remaining error splits into an irreducible part the features cannot
reach, and the cost of forecasting the market level 61 days ahead. Only the
second is addressable, which is why the level projection — not feature
engineering — is where any further work belongs.

## The best feature set

**The one already shipped.** No candidate earned inclusion.

```
distance, log_distance, straight_line, circuity, coords_unreliable
weight, weight_per_mile, miles_per_1k_lb
pickup_lat, pickup_lon, delivery_lat, delivery_lon
dow_sin, dow_cos, is_weekend
market_index, market_index_28d
route_freq
te_route, te_pickup, te_delivery, te_equipment_distance
equipment, pickup, delivery            (native categoricals)
```

Two honest caveats about that list, both carried over from earlier work:

- `te_route`, `te_delivery` and both `market_index` columns score near zero on
  permutation importance. They survive because they cost nothing and help the
  12.2% unseen-lane cases, not because they earn their place on the metric.
- `dow_sin` / `dow_cos` / `is_weekend` contribute about 0.05 points, which is
  inside noise. They are retained because they are the only thing that makes
  the December chart move — a presentation requirement, not an accuracy one.

A defensible alternative is to strip the list to `distance`, `log_distance`,
`weight`, `weight_per_mile`, `equipment`, `pickup`, `delivery`,
`te_equipment_distance` and the day-of-week terms. That is roughly ten columns
instead of twenty-four for materially the same accuracy. The current set is
kept because breadth costs nothing here and protects the unseen-lane cases.
