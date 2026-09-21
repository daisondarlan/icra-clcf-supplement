// Reimplements the fail-closed decision rule (paper Eq. 6-7 / Algorithm 2)
// on bare AVR hardware: a per-feature Gaussian Naive Bayes likelihood check,
// evaluated in log-space, plus the fail-closed out-of-range check. Parameters
// below are fit from the same post-fix dataset (Session E + the three
// controlled sessions) used for the paper's rebuilt likelihood tables --
// see ../arduino_gaussian_params.json for the source values.
//
// Protocol: host sends one line "angle_deg,flatness_std,obstacle_ratio\n" over
// serial; this replies "<0 or 1>,<microseconds>\n" (1 = SAFE, 0 = NOT SAFE).

const float MU_S[3] = {1.7718298f, 0.0250845f, 0.0584351f};
const float SD_S[3] = {2.4131501f, 0.0090053f, 0.0459337f};
const float MU_U[3] = {16.9270849f, 0.1253374f, 0.5092067f};
const float SD_U[3] = {9.7595147f, 0.1038350f, 0.2445976f};
const float LO[3]   = {0.0074498f, 0.0046347f, 0.0f};
const float HI[3]   = {32.1910445f, 2.2705456f, 0.8155258f};

float logGaussianPdf(float x, float mu, float sigma) {
  float z = (x - mu) / sigma;
  return -0.5f * z * z - logf(sigma) - 0.9189385332f;  // -0.5*ln(2*pi)
}

// 1 = SAFE, 0 = NOT SAFE
int classify(float x[3], unsigned long *us_taken) {
  unsigned long t0 = micros();
  for (int j = 0; j < 3; j++) {
    if (x[j] < LO[j] || x[j] > HI[j]) {
      *us_taken = micros() - t0;
      return 0;  // fail closed: out of range, no evidence either way
    }
  }
  float logp_safe = 0.0f, logp_unsafe = 0.0f;
  for (int j = 0; j < 3; j++) {
    logp_safe += logGaussianPdf(x[j], MU_S[j], SD_S[j]);
    logp_unsafe += logGaussianPdf(x[j], MU_U[j], SD_U[j]);
  }
  *us_taken = micros() - t0;
  return (logp_safe >= logp_unsafe) ? 1 : 0;
}

void setup() {
  Serial.begin(115200);
}

void loop() {
  if (Serial.available()) {
    String line = Serial.readStringUntil('\n');
    line.trim();
    if (line.length() == 0) return;

    char buf[64];
    line.toCharArray(buf, sizeof(buf));
    float x[3];
    int idx = 0;
    char *tok = strtok(buf, ",");
    while (tok != NULL && idx < 3) {
      x[idx++] = atof(tok);
      tok = strtok(NULL, ",");
    }

    if (idx == 3) {
      unsigned long us;
      int decision = classify(x, &us);
      Serial.print(decision);
      Serial.print(",");
      Serial.println(us);
    } else {
      Serial.println("ERR");
    }
  }
}
