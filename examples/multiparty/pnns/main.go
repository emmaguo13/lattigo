package main

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/multiparty"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
	"github.com/tuneinsight/lattigo/v6/utils/sampling"
)

// Demonstrates a minimal multiparty CKKS setup that calculates the cosine similarity
// between the embeddings of N parties, using Lattigo's built-in inner-sum
// rotations (PartialTracesSum via RotateAndAdd) rather than manual ring packing.
func main() {
	params, err := ckks.NewParametersFromLiteral(ckks.ExampleParameters128BitLogN14LogQP438)
	check(err)

	// N parties
	// Supports arbitrary N in principle, but practical limits related to performance and memory limit the actual size.
	// Stay in the low hundreds.
	parties := 3
	crs, err := sampling.NewKeyedPRNG([]byte("pnns-example"))
	check(err)

	// Generate secret key shares per party.
	kgen := rlwe.NewKeyGenerator(params)
	sks := make([]*rlwe.SecretKey, parties)
	for i := range sks {
		sks[i] = kgen.GenSecretKeyNew()
	}

	// Generate a collective public key.
	pk := genCollectivePK(params, crs, sks)
	// Generate the collective relinearization key. Allows you to shrink the ciphertext back to degree-1 after multiplication
	rlk := genCollectiveRelin(params, crs, sks)
	// Uses Galois elements for inner sum
	galEls := params.GaloisElementsForInnerSum(1, 120) // rotations for summing first 120 slots
	galKeys := genCollectiveRotations(params, crs, sks, galEls)

	evk := rlwe.NewMemEvaluationKeySet(rlk, galKeys...)

	encoder := ckks.NewEncoder(params)
	encryptor := rlwe.NewEncryptor(params, pk)

	// Deterministic test vectors so we can compare decrypted cosine to plaintext cosine.
	slotsToUse := 120
	if slotsToUse > params.MaxSlots() {
		slotsToUse = params.MaxSlots()
	}

	type vecCase struct {
		name      string
		aSeed     int64
		bSeed     int64
		scaleB    float64
		expectCos float64
	}

	cases := []vecCase{
		{name: "identical", aSeed: 1, bSeed: 1, scaleB: 1},      // cosine ~1
		{name: "opposite", aSeed: 2, bSeed: 2, scaleB: -1},      // cosine ~-1
		{name: "distinct", aSeed: 3, bSeed: 4, scaleB: 1},       // general position
		{name: "mixed-scale", aSeed: 5, bSeed: 6, scaleB: 0.42}, // scaled variant
	}

	eval := ckks.NewEvaluator(params, evk)
	aggSk := aggregateSecret(params, sks)
	decryptor := rlwe.NewDecryptor(params, aggSk)

	for _, tc := range cases {
		vecA := genVector(params.MaxSlots(), slotsToUse, tc.aSeed)
		vecB := scaleVector(genVector(params.MaxSlots(), slotsToUse, tc.bSeed), tc.scaleB)
		plainCos := cosine(vecA, vecB, slotsToUse)

		ptA := ckks.NewPlaintext(params, params.MaxLevel())
		ptB := ckks.NewPlaintext(params, params.MaxLevel())
		check(encoder.Encode(vecA, ptA))
		check(encoder.Encode(vecB, ptB))

		ctA, err := encryptor.EncryptNew(ptA)
		check(err)
		ctB, err := encryptor.EncryptNew(ptB)
		check(err)

		prod, err := eval.MulRelinNew(ctA, ctB)
		check(err)
		check(eval.Rescale(prod, prod))

		dot := ckks.NewCiphertext(params, 1, prod.Level())
		check(eval.RotateAndAdd(prod, 1, slotsToUse, dot)) // sum first slotsToUse slots into every slot

		normA, err := eval.MulRelinNew(ctA, ctA)
		check(err)
		check(eval.Rescale(normA, normA))
		check(eval.RotateAndAdd(normA, 1, slotsToUse, normA))

		normB, err := eval.MulRelinNew(ctB, ctB)
		check(err)
		check(eval.Rescale(normB, normB))
		check(eval.RotateAndAdd(normB, 1, slotsToUse, normB))

		decode := func(ct *rlwe.Ciphertext) float64 {
			pt := decryptor.DecryptNew(ct)
			out := make([]complex128, params.MaxSlots())
			check(encoder.Decode(pt, out))
			return real(out[0])
		}

		dotDec := decode(dot)
		normADec := decode(normA)
		normBDec := decode(normB)
		cosDec := dotDec / (math.Sqrt(normADec) * math.Sqrt(normBDec))

		fmt.Printf("[%s] cos(enc)=%.6f cos(plain)=%.6f diff=%.6f\n", tc.name, cosDec, plainCos, math.Abs(cosDec-plainCos))
	}

	fmt.Println("Cosine similarity can be obtained as dot / (sqrt(normA)*sqrt(normB)); results above compare encrypted vs plaintext.")
}

// genCollectivePK runs the CKG protocol.
func genCollectivePK(params ckks.Parameters, crs sampling.PRNG, sks []*rlwe.SecretKey) *rlwe.PublicKey {
	ckg := multiparty.NewPublicKeyGenProtocol(params)
	crp := ckg.SampleCRP(crs)

	shares := make([]multiparty.PublicKeyGenShare, len(sks))
	for i := range shares {
		shares[i] = ckg.AllocateShare()
		ckg.GenShare(sks[i], crp, &shares[i])
	}

	agg := ckg.AllocateShare()
	for i := range shares {
		ckg.AggregateShares(shares[i], agg, &agg)
	}

	pk := rlwe.NewPublicKey(params)
	ckg.GenPublicKey(agg, crp, pk)
	return pk
}

// genCollectiveRelin runs the two-round RKG protocol.
func genCollectiveRelin(params ckks.Parameters, crs sampling.PRNG, sks []*rlwe.SecretKey) *rlwe.RelinearizationKey {
	rkg := multiparty.NewRelinearizationKeyGenProtocol(params)
	crp := rkg.SampleCRP(crs)

	eph := make([]*rlwe.SecretKey, len(sks))
	r1 := make([]multiparty.RelinearizationKeyGenShare, len(sks))
	r2 := make([]multiparty.RelinearizationKeyGenShare, len(sks))
	for i := range sks {
		eph[i], r1[i], r2[i] = rkg.AllocateShare()
		rkg.GenShareRoundOne(sks[i], crp, eph[i], &r1[i])
	}

	_, r1Agg, r2Agg := rkg.AllocateShare()
	for i := range sks {
		rkg.AggregateShares(r1[i], r1Agg, &r1Agg)
	}
	for i := range sks {
		rkg.GenShareRoundTwo(eph[i], sks[i], r1Agg, &r2[i])
		rkg.AggregateShares(r2[i], r2Agg, &r2Agg)
	}

	rlk := rlwe.NewRelinearizationKey(params)
	rkg.GenRelinearizationKey(r1Agg, r2Agg, rlk)
	return rlk
}

// genCollectiveRotations runs the GKG protocol for a set of Galois elements.
func genCollectiveRotations(params ckks.Parameters, crs sampling.PRNG, sks []*rlwe.SecretKey, galEls []uint64) []*rlwe.GaloisKey {
	gkg := multiparty.NewGaloisKeyGenProtocol(params)

	galKeys := make([]*rlwe.GaloisKey, len(galEls))
	share := make([]multiparty.GaloisKeyGenShare, len(sks))
	for i := range share {
		share[i] = gkg.AllocateShare()
	}

	for j, galEl := range galEls {
		crp := gkg.SampleCRP(crs)
		for i := range sks {
			gkg.GenShare(sks[i], galEl, crp, &share[i])
		}

		agg := gkg.AllocateShare()
		agg.GaloisElement = galEl
		for i := range share {
			gkg.AggregateShares(share[i], agg, &agg)
		}

		galKeys[j] = rlwe.NewGaloisKey(params)
		check(gkg.GenGaloisKey(agg, crp, galKeys[j]))
	}
	return galKeys
}

// aggregateSecret adds the parties' secret shares into a single secret key.
func aggregateSecret(params ckks.Parameters, sks []*rlwe.SecretKey) *rlwe.SecretKey {
	sum := rlwe.NewSecretKey(params)
	r := params.RingQP()
	for i := range sks {
		r.Add(sum.Value, sks[i].Value, sum.Value)
	}
	return sum
}

func genVector(maxSlots, n int, seed int64) []complex128 {
	rng := rand.New(rand.NewSource(seed))
	v := make([]complex128, maxSlots)
	for i := 0; i < n && i < maxSlots; i++ {
		v[i] = complex(rng.NormFloat64(), 0)
	}
	return v
}

func scaleVector(in []complex128, scale float64) []complex128 {
	out := make([]complex128, len(in))
	for i, v := range in {
		out[i] = complex(scale*real(v), scale*imag(v))
	}
	return out
}

func cosine(a, b []complex128, n int) float64 {
	var dot, normA, normB float64
	for i := 0; i < n && i < len(a) && i < len(b); i++ {
		ra := real(a[i])
		rb := real(b[i])
		dot += ra * rb
		normA += ra * ra
		normB += rb * rb
	}
	return dot / (math.Sqrt(normA) * math.Sqrt(normB))
}

func check(err error) {
	if err != nil {
		panic(err)
	}
}
