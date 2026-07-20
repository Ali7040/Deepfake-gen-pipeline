export const vertexShader = /* glsl */ `
  varying vec2 vUv;

  void main() {
    vUv = uv;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

export const fragmentShader = /* glsl */ `
  precision highp float;

  uniform sampler2D uTexA;
  uniform sampler2D uTexB;
  uniform float uProgress;
  uniform float uTime;
  uniform vec2 uMouse;
  uniform float uHover;
  uniform float uAberration;
  uniform vec2 uResolution;
  uniform vec2 uImageSizeA;
  uniform vec2 uImageSizeB;

  varying vec2 vUv;

  float hash(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
  }

  float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    float a = hash(i);
    float b = hash(i + vec2(1.0, 0.0));
    float c = hash(i + vec2(0.0, 1.0));
    float d = hash(i + vec2(1.0, 1.0));
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(a, b, u.x) + (c - a) * u.y * (1.0 - u.x) + (d - b) * u.x * u.y;
  }

  vec2 coverUv(vec2 uv, vec2 imageSize, vec2 resolution) {
    vec2 s = resolution / imageSize;
    float minScale = min(s.x, s.y);
    float scale = clamp(max(s.x, s.y), minScale, minScale * 1.6);
    vec2 scaledImageSize = imageSize * scale;
    vec2 offset = (resolution - scaledImageSize) * vec2(0.5, 1.0);
    return (uv * resolution - offset) / scaledImageSize;
  }

  vec3 duotone(vec3 color, vec3 shadow, vec3 highlight) {
    float lum = dot(color, vec3(0.299, 0.587, 0.114));
    return mix(shadow, highlight, lum);
  }

  void main() {
    vec2 uvA = coverUv(vUv, uImageSizeA, uResolution);
    vec2 uvB = coverUv(vUv, uImageSizeB, uResolution);

    vec2 centered = vUv - 0.5;
    float dist = length(centered);

    vec2 mouseOffset = uMouse * 0.02 * uHover * (1.0 - dist);
    uvA += mouseOffset;
    uvB += mouseOffset;

    float n = noise(vUv * 5.0 + uTime * 0.03);
    float edge = 0.12;
    float mixFactor = smoothstep(n - edge, n + edge, uProgress);

    float band = 1.0 - abs(mixFactor - 0.5) * 2.0;
    float aberrationAmount = uAberration * (0.004 + band * 0.01);

    vec2 dir = centered * 2.0;
    float rA = texture2D(uTexA, uvA + dir * aberrationAmount).r;
    float gA = texture2D(uTexA, uvA).g;
    float bA = texture2D(uTexA, uvA - dir * aberrationAmount).b;
    vec3 colorA = vec3(rA, gA, bA);

    float rB = texture2D(uTexB, uvB + dir * aberrationAmount).r;
    float gB = texture2D(uTexB, uvB).g;
    float bB = texture2D(uTexB, uvB - dir * aberrationAmount).b;
    vec3 colorB = vec3(rB, gB, bB);

    vec3 baseColor = mix(colorA, colorB, mixFactor);

    vec3 shadow = vec3(0.01, 0.02, 0.035);
    vec3 highlightBlue = vec3(0.29, 0.70, 1.0);
    vec3 highlightWarm = vec3(1.0, 0.42, 0.21);
    vec3 highlight = mix(highlightBlue, highlightWarm, mixFactor * 0.4);
    vec3 toned = duotone(baseColor, shadow, highlight);

    vec3 finalColor = mix(baseColor, toned, 0.38);

    float vignette = smoothstep(0.95, 0.25, dist);
    finalColor *= mix(0.55, 1.0, vignette);

    float loopedTime = mod(uTime, 120.0);
    float grain = (hash(vUv * 900.0 + loopedTime * 37.0) - 0.5) * 0.035;
    finalColor += grain;

    gl_FragColor = vec4(finalColor, 1.0);
  }
`;
