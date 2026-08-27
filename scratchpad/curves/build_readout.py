import base64
import pathlib

HERE = pathlib.Path(__file__).parent
b64 = base64.b64encode((HERE / "overfit_localattn_curves.png").read_bytes()).decode()

ROWS = [
    ("train loss", "0.0153", "0.0144 <span class='ep'>@451</span>", "0.01515 &plusmn; 0.00037", "good", "flat"),
    ("validation loss", "0.1498", "0.0752 <span class='ep'>@63</span>", "0.1787 &plusmn; 0.0242", "bad", "rising"),
    ("single-step p@L &mdash; train", "0.850", "0.972 <span class='ep'>@468</span>", "0.913 &plusmn; 0.033", "good", "saturated"),
    ("single-step p@L &mdash; validation", "0.510", "0.931 <span class='ep'>@443</span>", "0.301 &plusmn; 0.168", "bad", "at floor"),
    ("single-step p@L &mdash; noisy floor", "0.334", "0.900 <span class='ep'>@443</span>", "0.344 &plusmn; 0.178", "muted", "baseline"),
    ("sampled p@L &mdash; mean", "0.3117", "0.3130 <span class='ep'>@475</span>", "0.3087 &plusmn; 0.0017", "warn", "plateau"),
    ("sampled p@L &mdash; median", "0.0652", "0.1213 <span class='ep'>@119</span>", "0.0655 &plusmn; 0.0042", "bad", "plateau, low"),
]

rows_html = "\n".join(
    f'      <tr><td class="metric">{name}</td>'
    f'<td class="num">{final}</td><td class="num">{best}</td>'
    f'<td class="num">{tail}</td>'
    f'<td><span class="pill {cls}">{verdict}</span></td></tr>'
    for name, final, best, tail, cls, verdict in ROWS
)

HTML = f"""<title>local_attn Convergence Readout</title>
<style>
  :root {{
    --ground: #FBFCFD; --surface: #F1F4F8; --surface-2: #E7EDF4; --line: #D4DDE7;
    --ink: #13181F; --ink-2: #48535F; --ink-3: #78838F;
    --blue: #2A6EBB; --good: #2C6A4C; --warn: #8A6110; --bad: #A03826;
    --good-bg: #E3F0E9; --warn-bg: #F6EBD4; --bad-bg: #F7E3DF; --muted-bg: #E7EDF4;
  }}
  @media (prefers-color-scheme: dark) {{
    :root:not([data-theme="light"]) {{
      --ground: #0E1218; --surface: #161C24; --surface-2: #1E252F; --line: #2C343F;
      --ink: #E5EBF2; --ink-2: #A7B3C1; --ink-3: #7B8794;
      --blue: #77ABE7; --good: #77C39B; --warn: #DBB05C; --bad: #E8866F;
      --good-bg: #172A22; --warn-bg: #2C2416; --bad-bg: #2E1D1A; --muted-bg: #1E252F;
    }}
  }}
  :root[data-theme="dark"] {{
    --ground: #0E1218; --surface: #161C24; --surface-2: #1E252F; --line: #2C343F;
    --ink: #E5EBF2; --ink-2: #A7B3C1; --ink-3: #7B8794;
    --blue: #77ABE7; --good: #77C39B; --warn: #DBB05C; --bad: #E8866F;
    --good-bg: #172A22; --warn-bg: #2C2416; --bad-bg: #2E1D1A; --muted-bg: #1E252F;
  }}

  body {{
    background: var(--ground);
    color: var(--ink);
    font-family: "Helvetica Neue", Helvetica, Arial, ui-sans-serif, system-ui, sans-serif;
    line-height: 1.55;
    margin: 0;
    padding: clamp(20px, 4vw, 52px);
  }}
  .wrap {{ max-width: 1180px; margin: 0 auto; display: flex; flex-direction: column; gap: 34px; }}

  header {{ display: flex; flex-direction: column; gap: 7px; }}
  .eyebrow {{
    font-size: 11.5px; letter-spacing: .13em; text-transform: uppercase;
    color: var(--ink-3); font-weight: 600;
  }}
  h1 {{ font-size: clamp(25px, 3.4vw, 35px); line-height: 1.14; margin: 0; letter-spacing: -.018em; text-wrap: balance; }}
  .sub {{ color: var(--ink-2); font-size: 15.5px; max-width: 68ch; margin: 0; }}

  h2 {{
    font-size: 12px; letter-spacing: .13em; text-transform: uppercase;
    color: var(--ink-3); font-weight: 600; margin: 0 0 13px 0;
  }}

  .verdicts {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(228px, 1fr)); gap: 13px; }}
  .card {{
    background: var(--surface); border: 1px solid var(--line);
    border-radius: 3px; padding: 15px 17px;
    display: flex; flex-direction: column; gap: 5px;
  }}
  .card .k {{ font-size: 11.5px; letter-spacing: .09em; text-transform: uppercase; color: var(--ink-3); font-weight: 600; }}
  .card .v {{
    font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
    font-size: 25px; font-variant-numeric: tabular-nums; letter-spacing: -.02em; line-height: 1.15;
  }}
  .card .n {{ font-size: 13.2px; color: var(--ink-2); }}
  .v.good {{ color: var(--good); }} .v.bad {{ color: var(--bad); }} .v.warn {{ color: var(--warn); }}

  figure {{ margin: 0; }}
  .figbox {{ background: #FFFFFF; border: 1px solid var(--line); border-radius: 3px; padding: 9px; overflow-x: auto; }}
  .figbox img {{ display: block; width: 100%; min-width: 720px; height: auto; }}
  figcaption {{ color: var(--ink-3); font-size: 13px; margin-top: 9px; max-width: 82ch; }}

  .tablebox {{ overflow-x: auto; border: 1px solid var(--line); border-radius: 3px; }}
  table {{ border-collapse: collapse; width: 100%; min-width: 660px; background: var(--surface); }}
  th, td {{ text-align: left; padding: 9px 15px; border-bottom: 1px solid var(--line); font-size: 14px; }}
  thead th {{
    background: var(--surface-2); font-size: 11px; letter-spacing: .09em;
    text-transform: uppercase; color: var(--ink-3); font-weight: 600; white-space: nowrap;
  }}
  tbody tr:last-child td {{ border-bottom: none; }}
  td.metric {{ color: var(--ink); }}
  td.num {{
    font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
    font-variant-numeric: tabular-nums; white-space: nowrap; color: var(--ink-2);
  }}
  .ep {{ color: var(--ink-3); font-size: 12px; }}
  .pill {{
    display: inline-block; padding: 1.5px 9px; border-radius: 2px;
    font-size: 11.5px; font-weight: 600; letter-spacing: .035em; white-space: nowrap;
  }}
  .pill.good {{ background: var(--good-bg); color: var(--good); }}
  .pill.bad {{ background: var(--bad-bg); color: var(--bad); }}
  .pill.warn {{ background: var(--warn-bg); color: var(--warn); }}
  .pill.muted {{ background: var(--muted-bg); color: var(--ink-3); }}

  .note {{
    border-left: 2px solid var(--blue); padding: 2px 0 2px 16px;
    color: var(--ink-2); font-size: 14.6px; max-width: 76ch;
    display: flex; flex-direction: column; gap: 9px;
  }}
  .note strong {{ color: var(--ink); font-weight: 600; }}
  .note p {{ margin: 0; }}

  .eta {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(196px, 1fr)); gap: 13px; }}
  code {{
    font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
    font-size: .9em; background: var(--surface-2); padding: 1px 5px; border-radius: 2px;
  }}
  footer {{ color: var(--ink-3); font-size: 12.5px; border-top: 1px solid var(--line); padding-top: 14px; }}
</style>

<div class="wrap">
  <header>
    <div class="eyebrow">Overfit-2000 A/B &middot; arm 1 of 3 &middot; 500/500 epochs</div>
    <h1>local_attn is fully converged on train and past its best on validation</h1>
    <p class="sub">Training loss went flat; validation loss bottomed out at <strong>epoch 63</strong> and has been
      climbing ever since. Full-sampling precision@L plateaued around epoch 300 and has not moved since.</p>
  </header>

  <section>
    <h2>The three readings that answer it</h2>
    <div class="verdicts">
      <div class="card">
        <div class="k">Train loss</div>
        <div class="v good">0.0152</div>
        <div class="n">last 20% of epochs, &plusmn;0.0004 &mdash; 2.4% relative spread. Flat.</div>
      </div>
      <div class="card">
        <div class="k">Validation loss</div>
        <div class="v bad">2.4&times;</div>
        <div class="n">worse than its epoch-63 best (0.0752 &rarr; 0.1787) and still drifting up.</div>
      </div>
      <div class="card">
        <div class="k">Sampled p@L, mean</div>
        <div class="v warn">0.309</div>
        <div class="n">&plusmn;0.002 over the last 100 epochs. A hard plateau, not a climb.</div>
      </div>
    </div>
  </section>

  <figure>
    <div class="figbox">
      <img src="data:image/png;base64,{b64}" alt="Three panels: log-scale train and validation loss over 500 epochs; single-step denoised precision@L for train, validation and the noisy-input floor; full-sampling precision@L mean and median on the fixed 32 validation chains.">
    </div>
    <figcaption>Left panel is log-scale so the validation rise is visible; at linear scale the epoch-10 spike flattens
      everything after it. Middle panel shows raw traces faintly with a 25-point rolling median on top &mdash; the raw
      single-step numbers swing with whichever noise level <em>t</em> each logged batch happened to draw.</figcaption>
  </figure>

  <section>
    <h2>Full numbers</h2>
    <div class="tablebox">
      <table>
        <thead>
          <tr><th>Metric</th><th>Final</th><th>Best (epoch)</th><th>Last 20% mean &plusmn; sd</th><th>Reading</th></tr>
        </thead>
        <tbody>
{rows_html}
        </tbody>
      </table>
    </div>
  </section>

  <section>
    <h2>What the validation panels actually say</h2>
    <div class="note">
      <p><strong>The single-step validation curve sits on its own noisy-input floor.</strong> Over the last 20% of
        training it averages 0.301 against a floor of 0.344 &mdash; so on held-out chains, one denoising step from a
        noised map recovers no more contacts than thresholding the noisy input itself. On train it reaches 0.913.
        That gap is the memorization, stated in the most interpretable metric available.</p>
      <p><strong>Mean and median sampled p@L disagree by 4.7&times;</strong> (0.309 vs 0.066). A mean far above the
        median means a handful of the 32 fixed chains score well and most score near zero &mdash; the arm has not
        learned a general map, it has learned a few. The median peaked at <strong>epoch 119</strong> and declined.</p>
      <p><strong>Best-checkpoint selection still works.</strong> The top-5 contact-precision checkpoints track the
        sampled mean, which is monotone to ~300 and flat after; nothing here invalidates the saved checkpoints, it
        just means epochs 300&ndash;500 bought nothing on validation.</p>
    </div>
  </section>

  <section>
    <h2>tri_mul ETA &mdash; measured, not projected from step counts</h2>
    <div class="eta">
      <div class="card">
        <div class="k">Per epoch</div>
        <div class="v">23.9 min</div>
        <div class="n">mean of four measured boundaries: 1354, 1349, 1675, 1362 s</div>
      </div>
      <div class="card">
        <div class="k">500 epochs</div>
        <div class="v bad">8.3 days</div>
        <div class="n">199 h of L40S wall-clock at batch 1</div>
      </div>
      <div class="card">
        <div class="k">Current chain covers</div>
        <div class="v warn">12%</div>
        <div class="n">4 &times; 6 h segments = 24 h of the 199 h needed</div>
      </div>
      <div class="card">
        <div class="k">vs conv_next</div>
        <div class="v">5.0&times;</div>
        <div class="n">slower per epoch (23.9 min vs 4.8 min)</div>
      </div>
    </div>
    <div class="note" style="margin-top:16px">
      <p><strong>Pure training is not the problem &mdash; batch 1 is.</strong> Measured at
        <code>0.74 s/minibatch</code>, steady across 85 windows. But batch 1 means 1802 minibatches per epoch against
        conv_next's 450, so the same 2000 chains cost 4&times; the launches at 8.6&times; the per-minibatch cost.
        Batch 1 was forced by OOM: batch 2 reached 43.28 GiB on a 44.39 GiB L40S.</p>
      <p>The 96 GB RTX PRO 6000 is the right card for this arm &mdash; it would take batch 4&ndash;8 and collapse the
        epoch count &mdash; which puts the <span style="color:var(--ink)">sm_120 environment on the critical path for
        tri_mul specifically</span>, not just as a nice-to-have.</p>
    </div>
  </section>

  <footer>
    Source: wandb run <code>overfit_localattn</code>, 11,413 history rows, all 500 epochs.
    ETA from job 20772472 log timestamps on node4103. Generated 2026-08-19.
  </footer>
</div>
"""

out = HERE / "localattn_readout.html"
out.write_text(HTML)
print("wrote", out, len(HTML), "bytes")
