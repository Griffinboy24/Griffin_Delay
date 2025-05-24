
/*
    XSimd_Helpers.h
*/


#pragma once

#include <complex>
#include <type_traits>
#include <cmath>

//==============================================================================
// 1. Sample-type meta helpers
//------------------------------------------------------------------------------
namespace chowdsp
{
    namespace SampleTypeHelpers
    {
        template <typename T, bool = std::is_floating_point_v<T>
        || std::is_same_v<T, std::complex<float>>
            || std::is_same_v<T, std::complex<double>>>
        struct TypeTraits
        {
            using ElementType = T;
            static constexpr int Size = 1;
        };

#if ! CHOWDSP_NO_XSIMD
        template <typename T>
        struct TypeTraits<xsimd::batch<T>, false>
        {
            using batch_type = xsimd::batch<T>;
            using ElementType = typename batch_type::value_type;
            static constexpr int Size = (int)batch_type::size;
        };

        template <typename T>
        struct TypeTraits<const xsimd::batch<T>, false>
        {
            using batch_type = xsimd::batch<T>;
            using ElementType = const typename batch_type::value_type;
            static constexpr int Size = (int)batch_type::size;
        };
#endif

        template <typename SampleType>
        using NumericType = typename TypeTraits<SampleType>::ElementType;

        template <typename ProcessorType>
        using ProcessorNumericType = typename ProcessorType::NumericType;

#if ! CHOWDSP_NO_XSIMD
        template <typename T,
            typename NumericType = NumericType<T>,
            typename SIMDType = xsimd::batch<NumericType>>
            inline constexpr bool IsSIMDRegister = std::is_same_v<T, SIMDType>;
#else
        template <typename> inline constexpr bool IsSIMDRegister = false;
#endif
    } // namespace SampleTypeHelpers
} // namespace chowdsp

//==============================================================================
// 2. Core SIMD-utility macros & default alignment
//------------------------------------------------------------------------------
#if ! CHOWDSP_NO_XSIMD
#define CHOWDSP_USING_XSIMD_STD(func) \
     using std::func;                  \
     using xsimd::func
#else
#define CHOWDSP_USING_XSIMD_STD(func) using std::func
#endif

namespace chowdsp::SIMDUtils
{
#if ! CHOWDSP_NO_XSIMD
    constexpr auto defaultSIMDAlignment = xsimd::default_arch::alignment();
#else
    constexpr size_t defaultSIMDAlignment = 16;
#endif
} // namespace chowdsp::SIMDUtils

//==============================================================================
// 3. Logic helpers (any/all/select)
//------------------------------------------------------------------------------
namespace chowdsp::SIMDUtils
{
#if ! CHOWDSP_NO_XSIMD
    template <typename VecBoolType>
    inline bool any(VecBoolType b) { return xsimd::any(b); }
    template <> inline bool any(bool b) { return b; }

    template <typename VecBoolType>
    inline bool all(VecBoolType b) { return xsimd::all(b); }
    template <> inline bool all(bool b) { return b; }

    template <typename T>
    inline T select(bool b, const T& t, const T& f) { return b ? t : f; }

    template <typename T>
    inline xsimd::batch<T> select(const xsimd::batch_bool<T>& b,
        const xsimd::batch<T>& t,
        const xsimd::batch<T>& f)
    {
        return xsimd::select(b, t, f);
    }
#else
    inline bool any(bool b) { return b; }
    inline bool all(bool b) { return b; }

    template <typename T>
    inline T select(bool b, const T& t, const T& f) { return b ? t : f; }
#endif
} // namespace chowdsp::SIMDUtils

//==============================================================================
// 4. Special maths (horizontal min/max, abs-max)
//------------------------------------------------------------------------------
namespace chowdsp::SIMDUtils
{
#if ! CHOWDSP_NO_XSIMD
    template <typename T>
    inline T hMaxSIMD(const xsimd::batch<T>& x)
    {
        constexpr auto N = xsimd::batch<T>::size;
        T v alignas (xsimd::default_arch::alignment())[N];
        xsimd::store_aligned(v, x);

        if constexpr (N == 2)
            return juce::jmax(v[0], v[1]);
        else if constexpr (N == 4)
            return juce::jmax(v[0], v[1], v[2], v[3]);
        else
            return juce::jmax(juce::jmax(v[0], v[1], v[2], v[3]),
                juce::jmax(v[4], v[5], v[6], v[7]));
    }

    template <typename T>
    inline T hMinSIMD(const xsimd::batch<T>& x)
    {
        constexpr auto N = xsimd::batch<T>::size;
        T v alignas (xsimd::default_arch::alignment())[N];
        xsimd::store_aligned(v, x);

        if constexpr (N == 2)
            return juce::jmin(v[0], v[1]);
        else if constexpr (N == 4)
            return juce::jmin(v[0], v[1], v[2], v[3]);
        else
            return juce::jmin(juce::jmin(v[0], v[1], v[2], v[3]),
                juce::jmin(v[4], v[5], v[6], v[7]));
    }

    template <typename T>
    inline T hAbsMaxSIMD(const xsimd::batch<T>& x) { return hMaxSIMD(xsimd::abs(x)); }
#endif
} // namespace chowdsp::SIMDUtils

//==============================================================================
// 5. Decibel helpers (scalar + SIMD)
//------------------------------------------------------------------------------
namespace chowdsp::SIMDUtils
{
    template <typename T>
    inline T gainToDecibels(T gain, T minusInfinityDB = (T)-100.0)
    {
        return juce::Decibels::gainToDecibels(gain, minusInfinityDB);
    }

    template <typename T>
    inline T decibelsToGain(T dB, T minusInfinityDB = (T)-100.0)
    {
        return juce::Decibels::decibelsToGain(dB, minusInfinityDB);
    }

#if ! CHOWDSP_NO_XSIMD
    template <typename T>
    inline xsimd::batch<T> gainToDecibels(const xsimd::batch<T>& gain,
        T minusInfinityDB = (T)-100.0)
    {
        using v = xsimd::batch<T>;
        return xsimd::select(gain > (T)0,
            xsimd::max(xsimd::log10(gain) * (T)20,
                (v)minusInfinityDB),
            (v)minusInfinityDB);
    }

    template <typename T>
    inline xsimd::batch<T> decibelsToGain(const xsimd::batch<T>& dB,
        T minusInfinityDB = (T)-100.0)
    {
        return xsimd::select(dB > minusInfinityDB,
            xsimd::pow((xsimd::batch<T>) 10,
                dB * (T)0.05),
            {});
    }
#endif
} // namespace chowdsp::SIMDUtils

//==============================================================================
// 6. Complex-math helpers (pow, polar, mul)
//------------------------------------------------------------------------------
namespace chowdsp::SIMDUtils
{
#if ! CHOWDSP_NO_XSIMD
    template <typename Type>
    inline xsimd::batch<Type> SIMDComplexMulReal(const xsimd::batch<std::complex<Type>>& a,
        const xsimd::batch<std::complex<Type>>& b)
    {
        return (a.real() * b.real()) - (a.imag() * b.imag());
    }

    template <typename Type>
    inline xsimd::batch<Type> SIMDComplexMulImag(const xsimd::batch<std::complex<Type>>& a,
        const xsimd::batch<std::complex<Type>>& b)
    {
        return (a.real() * b.imag()) + (a.imag() * b.real());
    }

    // --- pow overloads -----------------------------------------------------------
    template <typename BaseType, typename OtherType>
    inline std::enable_if_t<std::is_same_v<SampleTypeHelpers::NumericType<OtherType>,
        BaseType>,
        xsimd::batch<std::complex<BaseType>>>
        pow(const xsimd::batch<std::complex<BaseType>>& a, OtherType x)
    {
        auto absa = xsimd::abs(a);
        auto arga = xsimd::arg(a);
        auto r = xsimd::pow(absa, xsimd::batch(x));
        auto theta = x * arga;
        auto sc = xsimd::sincos(theta);
        return { r * sc.second, r * sc.first };
    }

    template <typename BaseType, typename OtherType>
    inline std::enable_if_t<std::is_same_v<SampleTypeHelpers::NumericType<OtherType>,
        BaseType>,
        xsimd::batch<std::complex<BaseType>>>
        pow(OtherType a, const xsimd::batch<std::complex<BaseType>>& z)
    {
        const auto ze = xsimd::batch((BaseType)0);
        auto absa = xsimd::abs(a);
        auto arga = xsimd::select(a >= ze, ze,
            xsimd::batch(juce::MathConstants<BaseType>::pi));
        auto x = z.real();
        auto y = z.imag();
        auto r = xsimd::pow(absa, x);
        auto theta = x * arga;

        auto cond = y == ze;
        r = select(cond, r, r * xsimd::exp(-y * arga));
        theta = select(cond, theta, theta + y * xsimd::log(absa));
        auto sc = xsimd::sincos(theta);
        return { r * sc.second, r * sc.first };
    }

    // --- polar -------------------------------------------------------------------
    template <typename BaseType>
    inline xsimd::batch<std::complex<BaseType>>
        polar(const xsimd::batch<BaseType>& mag,
            const xsimd::batch<BaseType>& angle)
    {
        auto sc = xsimd::sincos(angle);
        return { mag * sc.second, mag * sc.first };
    }

    template <typename BaseType>
    inline xsimd::batch<std::complex<BaseType>>
        polar(const xsimd::batch<BaseType>& angle)
    {
        auto sc = xsimd::sincos(angle);
        return { sc.second, sc.first };
    }
#endif // !CHOWDSP_NO_XSIMD
} // namespace chowdsp::SIMDUtils

//==============================================================================
// 7. Alignment helpers
//------------------------------------------------------------------------------
namespace chowdsp::SIMDUtils
{
    template <typename T>
    static bool isAligned(const T* p) noexcept
    {
#if ! CHOWDSP_NO_XSIMD
        constexpr auto RegSize = sizeof(xsimd::batch<T>);
#else
        constexpr auto RegSize = 16;
#endif
        uintptr_t mask = RegSize - 1;
        return ((uintptr_t)p & mask) == 0;
    }

    template <typename Type, typename IntType>
    inline Type* snapPointerToAlignment(Type* basePtr, IntType bytes) noexcept
    {
        return (Type*)((((size_t)basePtr) + (bytes - 1)) & ~(bytes - 1));
    }

    template <typename T>
    static T* getNextAlignedPtr(T* p) noexcept
    {
#if ! CHOWDSP_NO_XSIMD
        constexpr auto RegSize = sizeof(xsimd::batch<std::remove_const_t<T>>);
#else
        constexpr auto RegSize = 16;
#endif
        return snapPointerToAlignment(p, RegSize);
    }
} // namespace chowdsp::SIMDUtils

//==============================================================================
// 8. SIMD-aware SmoothedValue (CRTP over JUCE base)
//------------------------------------------------------------------------------
namespace chowdsp::SIMDUtils
{
    template <typename SmFloatType,
        typename SmoothingType>
    class SIMDSmoothedValue
        : public juce::SmoothedValueBase<SIMDSmoothedValue<
        xsimd::batch<SmFloatType>,
        SmoothingType>>
    {
    public:
        using VecType = xsimd::batch<SmFloatType>;

        SIMDSmoothedValue() noexcept
            : SIMDSmoothedValue((SmFloatType)(
                std::is_same_v<SmoothingType,
                juce::ValueSmoothingTypes::Linear>
                ? 0
                : 1))
        {
        }

        explicit SIMDSmoothedValue(SmFloatType initial) noexcept
        {
            jassert(!(std::is_same_v<SmoothingType,
                juce::ValueSmoothingTypes::Multiplicative>
                && initial == 0));

            this->currentValue = initial;
            this->target = this->currentValue;
        }

        //--------------------------------------------------------------------------
        void reset(double sampleRate,
            double seconds) noexcept
        {
            jassert(sampleRate > 0 && seconds >= 0);
            reset((int)std::floor(seconds * sampleRate));
        }

        void reset(int samples) noexcept
        {
            stepsToTarget = samples;
            this->setCurrentAndTargetValue(this->target);
        }

        //--------------------------------------------------------------------------
        void setTargetValue(VecType newTarget) noexcept
        {
            if (xsimd::all(newTarget == this->target))
                return;

            if (stepsToTarget <= 0)
            {
                this->setCurrentAndTargetValue(newTarget);
                return;
            }

            jassert(!(std::is_same_v<SmoothingType,
                juce::ValueSmoothingTypes::Multiplicative>
                && xsimd::any(newTarget == (SmFloatType)0)));

            this->target = newTarget;
            this->countdown = stepsToTarget;
            setStepSize();
        }

        //--------------------------------------------------------------------------
        VecType getNextValue() noexcept
        {
            if (!this->isSmoothing())
                return this->target;

            --(this->countdown);
            if (this->isSmoothing())
                setNextValue();
            else
                this->currentValue = this->target;

            return this->currentValue;
        }

        VecType skip(int numSamples) noexcept
        {
            if (numSamples >= this->countdown)
            {
                this->setCurrentAndTargetValue(this->target);
                return this->target;
            }

            skipCurrentValue(numSamples);
            this->countdown -= numSamples;
            return this->currentValue;
        }

    private:
#if CHOWDSP_USING_JUCE
        void applyGain(juce::AudioBuffer<SmFloatType>&, int) noexcept {}
#endif

        template <typename U = SmoothingType>
        std::enable_if_t<
            std::is_same_v<U, juce::ValueSmoothingTypes::Linear>, void>
            setStepSize() noexcept
        {
            step = (this->target - this->currentValue)
                / (VecType)(SmFloatType)this->countdown;
        }

        template <typename U = SmoothingType>
        std::enable_if_t<
            std::is_same_v<U, juce::ValueSmoothingTypes::Multiplicative>, void>
            setStepSize()
        {
            step = xsimd::exp((xsimd::log(xsimd::abs(this->target))
                - xsimd::log(xsimd::abs(this->currentValue)))
                / (VecType)(SmFloatType)this->countdown);
        }

        //--------------------------------------------------------------------------
        template <typename U = SmoothingType>
        std::enable_if_t<
            std::is_same_v<U, juce::ValueSmoothingTypes::Linear>, void>
            setNextValue() noexcept
        {
            this->currentValue += step;
        }

        template <typename U = SmoothingType>
        std::enable_if_t<
            std::is_same_v<U, juce::ValueSmoothingTypes::Multiplicative>, void>
            setNextValue() noexcept
        {
            this->currentValue *= step;
        }

        //--------------------------------------------------------------------------
        template <typename U = SmoothingType>
        std::enable_if_t<
            std::is_same_v<U, juce::ValueSmoothingTypes::Linear>, void>
            skipCurrentValue(int n) noexcept
        {
            this->currentValue += step * (SmFloatType)n;
        }

        template <typename U = SmoothingType>
        std::enable_if_t<
            std::is_same_v<U, juce::ValueSmoothingTypes::Multiplicative>, void>
            skipCurrentValue(int n)
        {
            this->currentValue *= xsimd::pow(step, (VecType)n);
        }

        //--------------------------------------------------------------------------
        VecType step = SmFloatType();
        int     stepsToTarget = 0;
    };
} // namespace chowdsp::SIMDUtils
