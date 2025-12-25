package main

import (
	"fmt"
	"math/rand"
	"time"

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
	// todo(emma): does lattigo support an unlimited N? 
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
	// todo(emma): what is collective relin? 
	rlk := genCollectiveRelin(params, crs, sks)
	// todo(emma): what are the galois elements for inner sum? 
	galEls := params.GaloisElementsForInnerSum(1, 120) // rotations for summing first 120 slots
	galKeys := genCollectiveRotations(params, crs, sks, galEls)

	evk := rlwe.NewMemEvaluationKeySet(rlk, galKeys...)

	encoder := ckks.NewEncoder(params)
	encryptor := rlwe.NewEncryptor(params, pk)

	// Toy embeddings: first 120 slots hold random values, rest are zero.
	rand.Seed(time.Now().UnixNano())
	fill := func() []complex128 {
		slots := params.MaxSlots()
		v := make([]complex128, slots)
		for i := 0; i < 120 && i < slots; i++ {
			v[i] = complex(rand.NormFloat64(), 0)
		}
		return v
	}

	// Generate two embeddings. Realistically we have an embedding per party member.
	ptA := ckks.NewPlaintext(params, params.MaxLevel())
	ptB := ckks.NewPlaintext(params, params.MaxLevel())
	check(encoder.Encode(fill(), ptA))
	check(encoder.Encode(fill(), ptB))

	ctA, err := encryptor.EncryptNew(ptA)
	check(err)
	ctB, err := encryptor.EncryptNew(ptB)
	check(err)

	eval := ckks.NewEvaluator(params, evk)

	// Dot product
	prod, err := eval.MulRelinNew(ctA, ctB)
	check(err)
	check(eval.Rescale(prod, prod))

	// todo(emma): make sure we are doing all of this with the shared public key.
	dot := ckks.NewCiphertext(params, 1, prod.Level())
	check(eval.RotateAndAdd(prod, 1, 120, dot)) // sum first 120 slots into every slot

	// Norms
	normA, err := eval.MulRelinNew(ctA, ctA)
	check(err)
	check(eval.Rescale(normA, normA))
	check(eval.RotateAndAdd(normA, 1, 120, normA))

	normB, err := eval.MulRelinNew(ctB, ctB)
	check(err)
	check(eval.Rescale(normB, normB))
	check(eval.RotateAndAdd(normB, 1, 120, normB))

	// For demo purposes we decrypt with the aggregated secret key (t-out-of-t).
	aggSk := aggregateSecret(params, sks)
	decryptor := rlwe.NewDecryptor(params, aggSk)

	printSlot := func(label string, ct *rlwe.Ciphertext) {
		pt := decryptor.DecryptNew(ct)
		out := make([]complex128, params.MaxSlots())
		check(encoder.Decode(pt, out))
		fmt.Printf("%s (slot 0): %.4f\n", label, real(out[0]))
	}

	printSlot("dot", dot)
	printSlot("normA", normA)
	printSlot("normB", normB)
	fmt.Println("Homomorphic cosine similarity can be obtained as dot / (sqrt(normA)*sqrt(normB)) either in the clear or via polynomial approx.")
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

func check(err error) {
	if err != nil {
		panic(err)
	}
}
