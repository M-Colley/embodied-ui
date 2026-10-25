using UnityEngine;
using UnityEngine.InputSystem;

namespace EmbodiedUI.PlayerTest
{
    /// <summary>
    /// The player scene's car: a rigid body accelerated by an Input System
    /// action bound to the embodied-ui device's vehicle.throttle control.
    /// Full throttle lights the lamp.
    /// </summary>
    [RequireComponent(typeof(Rigidbody))]
    public sealed class PlayerCar : MonoBehaviour
    {
        public Light lamp;
        public int fixedSteps;
        public float lastThrottle;
        InputAction throttle;
        Rigidbody body;

        void Awake()
        {
            body = GetComponent<Rigidbody>();
            body.useGravity = false;
            throttle = new InputAction("throttle", InputActionType.Value, "<EmbodiedUI>/vehicle_throttle");
            throttle.Enable();
        }

        void OnDestroy()
        {
            throttle.Disable();
            throttle.Dispose();
        }

        void FixedUpdate()
        {
            fixedSteps++;
            lastThrottle = throttle.ReadValue<float>();
            body.AddForce(transform.forward * 5f * lastThrottle, ForceMode.Acceleration);
            if (lamp != null) lamp.enabled = lastThrottle > 0.5f;
        }
    }
}
